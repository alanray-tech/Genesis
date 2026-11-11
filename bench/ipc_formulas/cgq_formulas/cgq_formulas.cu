// Reference side of the IPC formula parity: CGQ's contact functions (qipc/_src/native/solver/contact_function and the
// headers production composes them with), evaluated on rows of inputs, one kernel per case of cases.py.
//
// Each case composes the CGQ functions the way production does (consistent_ipc_contact.cu, contact_kernels.cu,
// halfplane_contact_kernels.cu, ccd_kernels.h): the distance flag and its dispatch to the per-primitive barrier, the
// mollifier dispatch of the edge-edge barrier, the rank-one fast path of the edge-edge barrier, the bound dispatch of
// the CCD. Constants are CGQ's
// production values. The rows a case reads and writes are documented in cases.py.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <solver/contact_function/candidate_reach.h>
#include <solver/contact_function/codim_thickness.h>
#include <solver/contact_function/directional_ccd.h>
#include <solver/contact_function/distance_flag.h>
#include <solver/contact_function/ee_mollified_barrier.h>
#include <solver/contact_function/friction_contact_function.h>
#include <solver/contact_function/gipc_accd.h>
#include <solver/contact_function/gipc_barrier.h>
#include <solver/contact_function/gipc_barrier_gradient_hessian.h>
#include <solver/contact_function/gipc_distance.h>
#include <solver/contact_function/gipc_distance_type.h>
#include <solver/contact_function/halfplane_contact.h>
#include <solver/contact_function/pair_d_hat.h>
#include <solver/contact_function/screw_ccd.h>
#include <solver/contact_function/screw_ccd_interval.h>
#include <solver/edge_triangle_intersect.h>

#include <string>
#include <unordered_map>

namespace qipc
{
struct HarnessTag;

// Fills the private pointers of a vertex context, through the friend hook CGQ keeps for its tests
template <>
class Test<HarnessTag>
{
  public:
    __device__ static void set(GlobalVertexContext& vtx,
                               double*              positions,
                               double*              ends,
                               double*              thicknesses,
                               double*              d_hats,
                               double*              inflations,
                               double*              speeds,
                               int64_t              n_verts)
    {
        vtx.positions_                = positions;
        vtx.trajectory_end_positions_ = ends;
        vtx.thicknesses_              = thicknesses;
        vtx.d_hats_                   = d_hats;
        vtx.path_inflation_           = inflations;
        vtx.path_speed_               = speeds;
        vtx.n_verts_                  = n_verts;
    }
};
}  // namespace qipc

namespace
{
using namespace qipc;

constexpr int32_t kBlock = 128;

struct Row
{
    const double* in;
    double*       out;
    __device__ double operator[](int i) const { return in[i]; }
    __device__ Vec3   v(int i) const { return Vec3(in[i], in[i + 1], in[i + 2]); }
    __device__ void   put(int i, double x) const { out[i] = x; }
    __device__ void   put(int i, const Vec3& x) const
    {
        out[i] = x(0);
        out[i + 1] = x(1);
        out[i + 2] = x(2);
    }
    template <int N>
    __device__ void put(int i, const Eigen::Matrix<double, N, 1>& x) const
    {
        for(int k = 0; k < N; ++k)
            out[i + k] = x(k);
    }
    template <int N>
    __device__ void put(int i, const Eigen::Matrix<double, N, N>& x) const
    {
        for(int r = 0; r < N; ++r)
            for(int c = 0; c < N; ++c)
                out[i + r * N + c] = x(r, c);
    }
};

using CaseFn = void (*)(const Row&);

__device__ uint32_t ee_type_flag(int code)
{
    constexpr uint32_t flags[9] = {0x5, 0x9, 0xD, 0x6, 0xA, 0xE, 0x7, 0xB, 0xF};
    return flags[code];
}

// --- Distances ---------------------------------------------------------------------------------------------------

__device__ void case_pt_distance(const Row& r)
{
    Vec3     verts[4] = {r.v(0), r.v(3), r.v(6), r.v(9)};
    uint32_t flag     = pt_distance_flag(verts[0], verts[1], verts[2], verts[3]);
    Vec12    G;
    pt_flagged_distance2_gradient(flag, verts, G);
    r.put(0, (double)flag);
    r.put(1, pt_flagged_distance2(flag, verts));
    r.put<12>(2, G);
}

__device__ void case_ee_distance(const Row& r)
{
    Vec3     verts[4] = {r.v(0), r.v(3), r.v(6), r.v(9)};
    uint32_t flag     = ee_distance_flag(verts[0], verts[1], verts[2], verts[3]);
    Vec12    G;
    ee_flagged_distance2_gradient(flag, verts, G);
    r.put(0, (double)flag);
    r.put(1, ee_flagged_distance2(flag, verts));
    r.put<12>(2, G);
}

__device__ void case_pe_distance(const Row& r)
{
    Vec3     verts[3] = {r.v(0), r.v(3), r.v(6)};
    uint32_t flag     = qipc::detail::point_edge_distance_flag_3(verts[0], verts[1], verts[2]);
    Vec9     G        = Vec9::Zero();
    double   d2;
    if(__popc(flag) == 3)
    {
        d2 = pe_distance2(verts[0], verts[1], verts[2]);
        pe_distance2_gradient(verts[0], verts[1], verts[2], G);
    }
    else
    {
        int32_t offs[3];
        flag_active_offsets(flag, offs);
        d2 = pp_distance2(verts[offs[0]], verts[offs[1]]);
        Vec6 G6;
        pp_distance2_gradient(verts[offs[0]], verts[offs[1]], G6);
        G.segment<3>(offs[0] * 3) = G6.segment<3>(0);
        G.segment<3>(offs[1] * 3) = G6.segment<3>(3);
    }
    r.put(0, (double)flag);
    r.put(1, d2);
    r.put<9>(2, G);
}

__device__ void case_ee_distance_type(const Row& r)
{
    r.put(0, (double)ee_type_flag(ee_distance_type(r.v(0), r.v(3), r.v(6), r.v(9))));
}

__device__ void case_closest_pt(const Row& r)
{
    Vec3   cb;
    double d = qipc::detail::closest_pt(r.v(0), r.v(3), r.v(6), r.v(9), cb);
    r.put(0, d);
    r.put(1, r.v(0));
    r.put(4, cb);
}

__device__ void case_closest_ee(const Row& r)
{
    Vec3   ca, cb;
    double d = qipc::detail::closest_ee(r.v(0), r.v(3), r.v(6), r.v(9), ca, cb);
    r.put(0, d);
    r.put(1, ca);
    r.put(4, cb);
}

__device__ void case_closest_pe(const Row& r)
{
    Vec3   cb;
    double d = qipc::detail::closest_pe(r.v(0), r.v(3), r.v(6), cb);
    r.put(0, d);
    r.put(1, r.v(0));
    r.put(4, cb);
}

// --- Barrier -----------------------------------------------------------------------------------------------------

__device__ void case_barrier(const Row& r)
{
    double D = r[0], d_hat = r[1], xi = r[2], kappa = r[3];
    r.put(0, gipc_barrier_energy(D, d_hat, xi, kappa));
    r.put(1, gipc_barrier_first_derivative(D, d_hat, xi, kappa));
    r.put(2, gipc_barrier_second_derivative(D, d_hat, xi, kappa));
    r.put(3, gipc_normal_force(D, d_hat, xi, kappa));
}

__device__ void case_mollified_barrier(const Row& r)
{
    r.put(0, gipc_barrier_energy_mollified(r[0], r[3], r[4], r[5], r[1], r[2]));
}

__device__ void case_mollifier(const Row& r)
{
    Vec3 p0 = r.v(0), p1 = r.v(3), p2 = r.v(6), p3 = r.v(9);
    r.put(0, gipc_compute_eps_x(p0, p1, p2, p3));
    r.put(1, (p1 - p0).cross(p3 - p2).squaredNorm());
}

template <int Kind>
__device__ void case_pair(const Row& r)
{
    // Kind: 0 PT, 1 EE, 2 PE, 3 PP; vertices of side A first
    constexpr int n_a = (Kind == 1) ? 2 : 1;
    constexpr int n   = (Kind == 0 || Kind == 1) ? 4 : (Kind == 2 ? 3 : 2);
    double        d_hats[4], thick[4];
    for(int i = 0; i < 4; ++i)
    {
        d_hats[i] = i < n_a ? r[0] : r[1];
        thick[i]  = r[2 + i];
    }
    BufferView<double> dh(d_hats, 0, n), th(thick, 0, n);
    double             d_hat, xi;
    if constexpr(Kind == 0)
    {
        d_hat = pair_d_hat_pt(dh, 0.0, 0, 1, 2, 3);
        xi    = pair_thickness_pt(th, 0, 1, 2, 3);
    }
    else if constexpr(Kind == 1)
    {
        d_hat = pair_d_hat_ee(dh, 0.0, 0, 1, 2, 3);
        xi    = pair_thickness_ee(th, 0, 1, 2, 3);
    }
    else if constexpr(Kind == 2)
    {
        d_hat = pair_d_hat_pe(dh, 0.0, 0, 1, 2);
        xi    = pair_thickness_pe(th, 0, 1, 2);
    }
    else
    {
        d_hat = pair_d_hat_pp(dh, 0.0, 0, 1);
        xi    = pair_thickness_pp(th, 0, 1);
    }
    double wa = r[6], wb = r[7];
    r.put(0, d_hat);
    r.put(1, xi);
    r.put(2, (wa < wb ? wa : wb) * d_hat / 4.0);  // cipc_pair_area_weight
}

template <int N>
__device__ void scatter(const Eigen::Matrix<double, 3 * N, 1>&      g,
                        const Eigen::Matrix<double, 3 * N, 3 * N>& h,
                        const int32_t*                              offs,
                        Vec12&                                      G,
                        Mat12&                                      H)
{
    G.setZero();
    H.setZero();
    for(int i = 0; i < N; ++i)
        G.segment<3>(offs[i] * 3) = g.template segment<3>(i * 3);
    for(int i = 0; i < N; ++i)
        for(int j = 0; j < N; ++j)
            H.block<3, 3>(offs[i] * 3, offs[j] * 3) = h.template block<3, 3>(i * 3, j * 3);
}

// cipc_filter_assemble_pt_kernel without the filter, the area weight and the scatter
__device__ void case_barrier_pt(const Row& r)
{
    Vec3     verts[4] = {r.v(0), r.v(3), r.v(6), r.v(9)};
    double   d_hat = r[12], xi = r[13], kappa = r[14];
    uint32_t flag = pt_distance_flag(verts[0], verts[1], verts[2], verts[3]);
    int32_t  pc   = __popc(flag);
    Vec12    G;
    Mat12    H;
    int32_t  offs[4];
    flag_active_offsets(flag, offs);
    if(pc == 4)
        gipc_barrier_grad_hess_pt(verts[0], verts[1], verts[2], verts[3], d_hat, xi, kappa, G, H);
    else if(pc == 3)
    {
        Vec9 g9;
        Mat9 h9;
        gipc_barrier_grad_hess_pe(verts[offs[0]], verts[offs[1]], verts[offs[2]], d_hat, xi, kappa, g9, h9);
        scatter<3>(g9, h9, offs, G, H);
    }
    else
    {
        Vec6 g6;
        Mat6 h6;
        gipc_barrier_grad_hess_pp(verts[offs[0]], verts[offs[1]], d_hat, xi, kappa, g6, h6);
        scatter<2>(g6, h6, offs, G, H);
    }
    r.put(0, (double)flag);
    r.put<12>(1, G);
    r.put<12>(13, H);
}

// cipc_filter_assemble_ee_kernel without the filter, the area weight and the scatter, outputs in stencil order
__device__ void case_barrier_ee(const Row& r)
{
    Vec3     verts[4] = {r.v(0), r.v(3), r.v(6), r.v(9)};
    Vec3     rest_v[4] = {r.v(12), r.v(15), r.v(18), r.v(21)};
    double   d_hat = r[24], xi = r[25], kappa = r[26];
    Vec3     p0 = verts[0], p1 = verts[1], p2 = verts[2], p3 = verts[3];
    uint32_t flag = ee_distance_flag(p0, p1, p2, p3);
    double   d2   = ee_flagged_distance2(flag, verts);
    int32_t  pc   = __popc(flag);
    double   I1    = (p1 - p0).cross(p3 - p2).squaredNorm();
    double   eps_x = gipc_compute_eps_x(rest_v[0], rest_v[1], rest_v[2], rest_v[3]);
    Vec12    G;
    Mat12    H;
    bool     is_mollified = I1 < eps_x;
    if(is_mollified)
    {
        double D = ee_mollified_distance2(p0, p1, p2, p3, flag, d2, xi, d_hat);
        ee_mollified_barrier_grad_hess(verts, flag, D, eps_x, d_hat, xi, kappa, G, H);
    }
    else if(pc == 4)
    {
        double v[12], grad_scale, hess_coef;
        gipc_barrier_grad_hess_ee_rank1(p0, p1, p2, p3, d_hat, xi, kappa, v, grad_scale, hess_coef);
        for(int i = 0; i < 12; ++i)
        {
            G(i) = v[i] * grad_scale;
            for(int j = 0; j < 12; ++j)
                H(i, j) = hess_coef * v[i] * v[j];
        }
    }
    else
    {
        int32_t offs[4];
        if(pc == 3)
        {
            bool is_pt_on_A;
            flag_active_offsets_for_ee(flag, offs, is_pt_on_A);
            Vec9 g9;
            Mat9 h9;
            gipc_barrier_grad_hess_pe(verts[offs[0]], verts[offs[1]], verts[offs[2]], d_hat, xi, kappa, g9, h9);
            scatter<3>(g9, h9, offs, G, H);
        }
        else
        {
            flag_active_offsets(flag, offs);
            Vec6 g6;
            Mat6 h6;
            gipc_barrier_grad_hess_pp(verts[offs[0]], verts[offs[1]], d_hat, xi, kappa, g6, h6);
            scatter<2>(g6, h6, offs, G, H);
        }
    }
    r.put(0, (double)flag);
    r.put(1, is_mollified ? 1.0 : 0.0);
    r.put<12>(2, G);
    r.put<12>(14, H);
}

// cipc_filter_assemble_pe_kernel without the filter, the area weight and the scatter
__device__ void case_barrier_pe(const Row& r)
{
    Vec3     verts[3] = {r.v(0), r.v(3), r.v(6)};
    double   d_hat = r[9], xi = r[10], kappa = r[11];
    uint32_t flag = qipc::detail::point_edge_distance_flag_3(verts[0], verts[1], verts[2]);
    Vec9     G    = Vec9::Zero();
    Mat9     H    = Mat9::Zero();
    if(__popc(flag) == 3)
        gipc_barrier_grad_hess_pe(verts[0], verts[1], verts[2], d_hat, xi, kappa, G, H);
    else
    {
        int32_t offs[3];
        flag_active_offsets(flag, offs);
        Vec6 g6;
        Mat6 h6;
        gipc_barrier_grad_hess_pp(verts[offs[0]], verts[offs[1]], d_hat, xi, kappa, g6, h6);
        for(int i = 0; i < 2; ++i)
        {
            G.segment<3>(offs[i] * 3) = g6.segment<3>(i * 3);
            for(int j = 0; j < 2; ++j)
                H.block<3, 3>(offs[i] * 3, offs[j] * 3) = h6.block<3, 3>(i * 3, j * 3);
        }
    }
    r.put(0, (double)flag);
    r.put<9>(1, G);
    r.put<9>(10, H);
}

__device__ void case_barrier_pp(const Row& r)
{
    Vec6 g;
    Mat6 h;
    gipc_barrier_grad_hess_pp(r.v(0), r.v(3), r[6], r[7], r[8], g, h);
    r.put<6>(0, g);
    r.put<6>(6, h);
}

// cipc_halfplane_filter_assemble_kernel without the filter, the area weight and the scatter
__device__ void case_barrier_ph(const Row& r)
{
    Vec3   x = r.v(0), P = r.v(3), N = r.v(6);
    double d_hat = r[9], xi = r[10], kappa = r[11];
    double d    = halfplane_signed_distance(x, P, N);
    double d_sq = d * d;
    double g_b  = gipc_barrier_first_derivative(d_sq, d_hat, xi, kappa);
    double H_b  = gipc_barrier_second_derivative(d_sq, d_hat, xi, kappa);
    r.put(0, halfplane_barrier_gradient(g_b, d, N));
    Eigen::Matrix3d h = halfplane_barrier_hessian(H_b, g_b, d_sq, N);
    r.put<3>(3, h);
}

// --- Friction ----------------------------------------------------------------------------------------------------

__device__ void case_friction_pt(const Row& r)
{
    double d_hat = r[24], kappa = r[25], mu = r[26], eps_vh = r[27], xi = r[28];
    Vec12  G;
    Mat12  H;
    r.put(0,
          gipc_friction_energy_pt(
              r.v(0), r.v(3), r.v(6), r.v(9), r.v(12), r.v(15), r.v(18), r.v(21), d_hat * d_hat, kappa, mu, eps_vh, xi));
    gipc_friction_grad_hess_pt(
        r.v(0), r.v(3), r.v(6), r.v(9), r.v(12), r.v(15), r.v(18), r.v(21), d_hat * d_hat, kappa, mu, eps_vh, G, H, xi);
    r.put<12>(1, G);
    r.put<12>(13, H);
}

__device__ void case_friction_ee(const Row& r)
{
    double d_hat = r[24], kappa = r[25], mu = r[26], eps_vh = r[27], xi = r[28];
    Vec12  G;
    Mat12  H;
    r.put(0,
          gipc_friction_energy_ee(
              r.v(0), r.v(3), r.v(6), r.v(9), r.v(12), r.v(15), r.v(18), r.v(21), d_hat * d_hat, kappa, mu, eps_vh, xi));
    gipc_friction_grad_hess_ee(
        r.v(0), r.v(3), r.v(6), r.v(9), r.v(12), r.v(15), r.v(18), r.v(21), d_hat * d_hat, kappa, mu, eps_vh, G, H, xi);
    r.put<12>(1, G);
    r.put<12>(13, H);
}

__device__ void case_friction_pe(const Row& r)
{
    double d_hat = r[18], kappa = r[19], mu = r[20], eps_vh = r[21], xi = r[22];
    Vec9   G;
    Mat9   H;
    r.put(0,
          gipc_friction_energy_pe(r.v(0), r.v(3), r.v(6), r.v(9), r.v(12), r.v(15), d_hat * d_hat, kappa, mu, eps_vh, xi));
    gipc_friction_grad_hess_pe(r.v(0), r.v(3), r.v(6), r.v(9), r.v(12), r.v(15), d_hat * d_hat, kappa, mu, eps_vh, G, H, xi);
    r.put<9>(1, G);
    r.put<9>(10, H);
}

__device__ void case_friction_pp(const Row& r)
{
    double d_hat = r[12], kappa = r[13], mu = r[14], eps_vh = r[15], xi = r[16];
    Vec6   G;
    Mat6   H;
    r.put(0, gipc_friction_energy_pp(r.v(0), r.v(3), r.v(6), r.v(9), d_hat * d_hat, kappa, mu, eps_vh, xi));
    gipc_friction_grad_hess_pp(r.v(0), r.v(3), r.v(6), r.v(9), d_hat * d_hat, kappa, mu, eps_vh, G, H, xi);
    r.put<6>(1, G);
    r.put<6>(7, H);
}

__device__ void case_friction_ph(const Row& r)
{
    double          d_hat = r[12], kappa = r[13], mu = r[14], eps_vh = r[15], xi = r[16];
    Eigen::Vector3d G;
    Eigen::Matrix3d H;
    r.put(0, halfplane_friction_energy(r.v(0), r.v(3), r.v(6), r.v(9), d_hat * d_hat, kappa, mu, eps_vh, xi));
    halfplane_friction_grad_hess(r.v(0), r.v(3), r.v(6), r.v(9), d_hat * d_hat, kappa, mu, eps_vh, G, H, xi);
    r.put(1, G);
    r.put<3>(4, H);
}

__device__ void put_basis(const Row& r, const Mat3x2& basis)
{
    r.put(0, Vec3(basis.col(0)));
    r.put(3, Vec3(basis.col(1)));
}

__device__ void put_projector(const Row& r, const Mat3x2& basis)
{
    Eigen::Matrix3d proj = basis * basis.transpose();
    r.put<3>(0, proj);
}

__device__ void case_frame_pt(const Row& r)
{
    Mat3x2 basis;
    Vec2   beta;
    pt_friction_tangent_basis(r.v(0), r.v(3), r.v(6), r.v(9), basis);
    pt_friction_closest_point(r.v(0), r.v(3), r.v(6), r.v(9), beta);
    put_basis(r, basis);
    r.put(6, 1.0);
    r.put(7, -1.0 + beta[0] + beta[1]);
    r.put(8, -beta[0]);
    r.put(9, -beta[1]);
}

__device__ void case_frame_ee(const Row& r)
{
    Mat3x2 basis;
    Vec2   gamma;
    ee_friction_tangent_basis(r.v(0), r.v(3), r.v(6), r.v(9), basis);
    ee_friction_closest_point(r.v(0), r.v(3), r.v(6), r.v(9), gamma);
    put_basis(r, basis);
    r.put(6, 1.0 - gamma[0]);
    r.put(7, gamma[0]);
    r.put(8, gamma[1] - 1.0);
    r.put(9, -gamma[1]);
}

__device__ void case_frame_pe(const Row& r)
{
    Mat3x2 basis;
    double eta;
    pe_friction_tangent_basis(r.v(0), r.v(3), r.v(6), basis);
    pe_friction_closest_point(r.v(0), r.v(3), r.v(6), eta);
    put_basis(r, basis);
    r.put(6, 1.0);
    r.put(7, eta - 1.0);
    r.put(8, -eta);
}

__device__ void case_frame_pp(const Row& r)
{
    Mat3x2 basis;
    pp_friction_tangent_basis(r.v(0), r.v(3), basis);
    put_projector(r, basis);
}

__device__ void case_frame_ph(const Row& r)
{
    Mat3x2 basis;
    halfplane_friction_tangent_basis(r.v(0), basis);
    put_projector(r, basis);
}

// --- CCD ---------------------------------------------------------------------------------------------------------

constexpr int kPT = 0, kEE = 1, kPE = 2, kPP = 3;

template <int Kind>
constexpr int n_verts_of()
{
    return Kind == kPP ? 2 : (Kind == kPE ? 3 : 4);
}

template <int Kind, bool Directional>
__device__ void case_ccd_linear(const Row& r)
{
    constexpr int n   = n_verts_of<Kind>();
    double        eta = r[6 * n], xi = r[6 * n + 1];
    Vec3          x[4], d[4];
    for(int i = 0; i < n; ++i)
    {
        x[i] = r.v(3 * i);
        d[i] = r.v(3 * n + 3 * i);
    }
    double toi;
    if constexpr(Kind == kPT)
        toi = Directional ? directional_point_triangle_ccd(x[0], x[1], x[2], x[3], d[0], d[1], d[2], d[3], eta, xi) :
                            gipc_point_triangle_ccd<false>(x[0], x[1], x[2], x[3], d[0], d[1], d[2], d[3], eta, xi);
    else if constexpr(Kind == kEE)
        toi = Directional ? directional_edge_edge_ccd(x[0], x[1], x[2], x[3], d[0], d[1], d[2], d[3], eta, xi) :
                            gipc_edge_edge_ccd<false>(x[0], x[1], x[2], x[3], d[0], d[1], d[2], d[3], eta, xi);
    else if constexpr(Kind == kPE)
        toi = Directional ? directional_point_edge_ccd(x[0], x[1], x[2], d[0], d[1], d[2], eta, xi) :
                            gipc_point_edge_ccd(x[0], x[1], x[2], d[0], d[1], d[2], eta, xi);
    else
        toi = Directional ? directional_point_point_ccd(x[0], x[1], d[0], d[1], eta, xi) :
                            gipc_point_point_ccd(x[0], x[1], d[0], d[1], eta, xi);
    r.put(0, toi);
}

// ccd_screw_vert: a vertex with a zero rotation keeps its linear path
__device__ ScrewVert screw_vert(const Vec3& x0, const Vec3& d, const Vec3& w, const Vec3& piv, const Vec3& pd)
{
    return w.squaredNorm() > 0.0 ? ScrewVert::make(x0, d, w, piv, pd) : ScrewVert::make_linear(x0, d);
}

// Rows: points (3n), chords (3n), rotation, pivot, pivot displacement of side A then side B (18), eta, xi,
// is_same_rigid_motion (read by the Genesis side alone) [, t_start, t_end, distance_start, max_iters]
template <int Kind>
__device__ void load_screw(const Row& r, ScrewVert (&v)[4])
{
    constexpr int n   = n_verts_of<Kind>();
    constexpr int n_a = Kind == kEE ? 2 : 1;
    for(int i = 0; i < n; ++i)
    {
        int side = i < n_a ? 0 : 1;
        v[i]     = screw_vert(r.v(3 * i), r.v(3 * n + 3 * i), r.v(6 * n + 9 * side), r.v(6 * n + 9 * side + 3), r.v(6 * n + 9 * side + 6));
    }
}

template <int Kind>
__device__ void case_ccd_screw(const Row& r)
{
    constexpr int n   = n_verts_of<Kind>();
    double        eta = r[6 * n + 18], xi = r[6 * n + 19];
    ScrewVert     v[4];
    load_screw<Kind>(r, v);
    double toi;
    if constexpr(Kind == kPT)
        toi = screw_point_triangle_ccd(v[0], v[1], v[2], v[3], eta, xi);
    else if constexpr(Kind == kEE)
        toi = screw_edge_edge_ccd(v[0], v[1], v[2], v[3], eta, xi);
    else if constexpr(Kind == kPE)
        toi = screw_point_edge_ccd(v[0], v[1], v[2], eta, xi);
    else
        toi = screw_point_point_ccd(v[0], v[1], eta, xi);
    r.put(0, toi);
}

template <int Kind>
__device__ void case_ccd_screw_interval(const Row& r)
{
    constexpr int n   = n_verts_of<Kind>();
    double        eta = r[6 * n + 18], xi = r[6 * n + 19];
    double        a = r[6 * n + 21], b = r[6 * n + 22], d0 = r[6 * n + 23];
    int32_t       max_iters = (int32_t)r[6 * n + 24];
    ScrewVert     v[4];
    load_screw<Kind>(r, v);
    ScrewCCDStep step;
    if constexpr(Kind == kPT)
        step = screw_point_triangle_ccd_interval(v[0], v[1], v[2], v[3], a, b, d0, eta, xi, max_iters);
    else
        step = screw_edge_edge_ccd_interval(v[0], v[1], v[2], v[3], a, b, d0, eta, xi, max_iters);
    r.put(0, step.t);
    r.put(1, (double)step.iters);
    r.put(2, (double)(int32_t)step.status);
}

// The linear branch of ccd_ph_pair_toi (ccd_kernels.h) for a vertex of no screw or articulated path
__device__ void case_ccd_ph(const Row& r)
{
    Vec3   x = r.v(0), dx = r.v(3), P = r.v(6), N = r.v(9);
    double eta = r[12], xi = r[13];
    double current_dist   = N.dot(x - P);
    double safe_dist      = current_dist - xi;
    double approach_speed = -N.dot(dx);
    bool   bound_active   = approach_speed > 0.0;
    r.put(0, bound_active ? safe_dist / approach_speed * (1.0 - eta) : 1.0);
    r.put(1, bound_active ? 1.0 : 0.0);
}

__device__ void case_ccd_ph_screw(const Row& r)
{
    Vec3      P = r.v(15), N = r.v(18);
    ScrewVert V = ScrewVert::make(r.v(0), r.v(3), r.v(6), r.v(9), r.v(12));
    r.put(0, screw_halfplane_ccd(V, N, N.dot(P), r[21], r[22]));
}

// --- Broad-phase distance rule -----------------------------------------------------------------------------------

__device__ const GlobalContactContext* g_cctx = nullptr;

// Rows: points (3n), ends (3n), inflations (n), speeds (n), thicknesses (n), d_hat, directional
template <int Kind>
__device__ void case_candidate(const Row& r)
{
    constexpr int n = n_verts_of<Kind>();
    double        x[12], e[12], infl[4], speed[4], thick[4], d_hats[4];
    for(int i = 0; i < 3 * n; ++i)
    {
        x[i] = r[i];
        e[i] = r[3 * n + i];
    }
    for(int i = 0; i < n; ++i)
    {
        infl[i]   = r[6 * n + i];
        speed[i]  = r[7 * n + i];
        thick[i]  = r[8 * n + i];
        d_hats[i] = r[9 * n];
    }
    bool                directional = r[9 * n + 1] != 0.0;
    GlobalVertexContext vtx;
    Test<HarnessTag>::set(vtx, x, e, thick, d_hats, infl, speed, n);
    CandidateReach verdict;
    if constexpr(Kind == kPT)
        verdict = pt_candidate_reach(vtx, *g_cctx, 0, 1, 2, 3, directional);
    else if constexpr(Kind == kEE)
        verdict = ee_candidate_reach(vtx, *g_cctx, 0, 1, 2, 3, directional);
    else if constexpr(Kind == kPE)
        verdict = pe_candidate_reach(vtx, *g_cctx, 0, 1, 2, directional);
    else
        verdict = pp_candidate_reach(vtx, *g_cctx, 0, 1, directional);
    r.put(0, (double)(int32_t)verdict);
}

// The cull of halfplane_query_kernel (halfplane_contact_kernels.cu), whose lines live in the kernel body.
// Rows: x, end, plane point, unit normal, inflation, d_hat, xi
__device__ void case_candidate_ph(const Row& r)
{
    Vec3   x = r.v(0), P = r.v(6), N = r.v(9);
    Vec3   dx = r.v(3) - x;
    double infl = r[12], d_hat = r[13], xi = r[14];
    double d     = halfplane_signed_distance(x, P, N);
    double d_dx  = N.dot(dx);
    double d_min = d + fmin(0.0, d_dx);
    double d_max = d + fmax(0.0, d_dx);
    if(infl > 0.0)
    {
        d_min -= infl;
        d_max += infl;
    }
    r.put(0, (d_max <= 0.0 || d_min >= d_hat * 1.0 + xi) ? 0.0 : 1.0);
}

// --- Intersection ------------------------------------------------------------------------------------------------

__device__ void case_intersect(const Row& r)
{
    auto d3 = [&](int i) { return make_double3(r[i], r[i + 1], r[i + 2]); };
    r.put(0, tri_edge_intersect(d3(0), d3(3), d3(6), d3(9), d3(12)) ? 1.0 : 0.0);
}

// --- Dispatch ----------------------------------------------------------------------------------------------------

template <CaseFn F>
__global__ void run_kernel(int64_t n, const double* in, int64_t in_stride, double* out, int64_t out_stride)
{
    for(int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += (int64_t)blockDim.x * gridDim.x)
        F(Row{in + i * in_stride, out + i * out_stride});
}

using Launcher = void (*)(int64_t, const double*, int64_t, double*, int64_t, cudaStream_t);

template <CaseFn F>
void launch(int64_t n, const double* in, int64_t in_stride, double* out, int64_t out_stride, cudaStream_t stream)
{
    int64_t blocks = std::min<int64_t>((n + kBlock - 1) / kBlock, 65535);
    if(blocks > 0)
        run_kernel<F><<<blocks, kBlock, 0, stream>>>(n, in, in_stride, out, out_stride);
}

const std::unordered_map<std::string, Launcher>& launchers()
{
    static const std::unordered_map<std::string, Launcher> table = {
        {"pt_distance", launch<case_pt_distance>},
        {"ee_distance", launch<case_ee_distance>},
        {"pe_distance", launch<case_pe_distance>},
        {"ee_distance_type", launch<case_ee_distance_type>},
        {"closest_pt", launch<case_closest_pt>},
        {"closest_ee", launch<case_closest_ee>},
        {"closest_pe", launch<case_closest_pe>},
        {"barrier", launch<case_barrier>},
        {"mollified_barrier", launch<case_mollified_barrier>},
        {"mollifier", launch<case_mollifier>},
        {"pair_pt", launch<case_pair<0>>},
        {"pair_ee", launch<case_pair<1>>},
        {"pair_pe", launch<case_pair<2>>},
        {"pair_pp", launch<case_pair<3>>},
        {"barrier_pt", launch<case_barrier_pt>},
        {"barrier_ee", launch<case_barrier_ee>},
        {"barrier_pe", launch<case_barrier_pe>},
        {"barrier_pp", launch<case_barrier_pp>},
        {"barrier_ph", launch<case_barrier_ph>},
        {"friction_pt", launch<case_friction_pt>},
        {"friction_ee", launch<case_friction_ee>},
        {"friction_pe", launch<case_friction_pe>},
        {"friction_pp", launch<case_friction_pp>},
        {"friction_ph", launch<case_friction_ph>},
        {"frame_pt", launch<case_frame_pt>},
        {"frame_ee", launch<case_frame_ee>},
        {"frame_pe", launch<case_frame_pe>},
        {"frame_pp", launch<case_frame_pp>},
        {"frame_ph", launch<case_frame_ph>},
        {"ccd_additive_pt", launch<case_ccd_linear<kPT, false>>},
        {"ccd_additive_ee", launch<case_ccd_linear<kEE, false>>},
        {"ccd_additive_pe", launch<case_ccd_linear<kPE, false>>},
        {"ccd_additive_pp", launch<case_ccd_linear<kPP, false>>},
        {"ccd_directional_pt", launch<case_ccd_linear<kPT, true>>},
        {"ccd_directional_ee", launch<case_ccd_linear<kEE, true>>},
        {"ccd_directional_pe", launch<case_ccd_linear<kPE, true>>},
        {"ccd_directional_pp", launch<case_ccd_linear<kPP, true>>},
        {"ccd_screw_pt", launch<case_ccd_screw<kPT>>},
        {"ccd_screw_ee", launch<case_ccd_screw<kEE>>},
        {"ccd_screw_pe", launch<case_ccd_screw<kPE>>},
        {"ccd_screw_pp", launch<case_ccd_screw<kPP>>},
        {"ccd_screw_interval_pt", launch<case_ccd_screw_interval<kPT>>},
        {"ccd_screw_interval_ee", launch<case_ccd_screw_interval<kEE>>},
        {"ccd_ph", launch<case_ccd_ph>},
        {"ccd_ph_screw", launch<case_ccd_ph_screw>},
        {"candidate_pt", launch<case_candidate<kPT>>},
        {"candidate_ee", launch<case_candidate<kEE>>},
        {"candidate_pe", launch<case_candidate<kPE>>},
        {"candidate_pp", launch<case_candidate<kPP>>},
        {"candidate_ph", launch<case_candidate_ph>},
        {"intersect", launch<case_intersect>},
    };
    return table;
}

GlobalContactContext* g_cctx_device = nullptr;

void set_candidate_context(double d_hat, double reach_scale)
{
    GlobalContactContext host;
    host.d_hat                 = d_hat;
    host.candidate_reach_scale = reach_scale;
    if(!g_cctx_device)
        TORCH_CHECK(cudaMalloc(&g_cctx_device, sizeof(GlobalContactContext)) == cudaSuccess);
    TORCH_CHECK(cudaMemcpy(g_cctx_device, &host, sizeof(GlobalContactContext), cudaMemcpyHostToDevice) == cudaSuccess);
    const GlobalContactContext* ptr = g_cctx_device;
    TORCH_CHECK(cudaMemcpyToSymbol(g_cctx, &ptr, sizeof(ptr)) == cudaSuccess);
}

void run(const std::string& name, torch::Tensor in, torch::Tensor out)
{
    TORCH_CHECK(in.is_cuda() && out.is_cuda() && in.dtype() == torch::kFloat64 && out.dtype() == torch::kFloat64);
    TORCH_CHECK(in.is_contiguous() && out.is_contiguous() && in.size(0) == out.size(0));
    auto it = launchers().find(name);
    TORCH_CHECK(it != launchers().end(), "unknown case ", name);
    it->second(in.size(0), in.data_ptr<double>(), in.size(1), out.data_ptr<double>(), out.size(1),
               at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
}

std::vector<std::string> case_names()
{
    std::vector<std::string> names;
    for(const auto& [name, _] : launchers())
        names.push_back(name);
    return names;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("run", &run);
    m.def("set_candidate_context", &set_candidate_context);
    m.def("case_names", &case_names);
}
