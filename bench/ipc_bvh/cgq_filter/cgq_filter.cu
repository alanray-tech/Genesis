// CGQ's classification-only narrow phase, compiled from CGQ's own classifiers (contact_function/distance_flag.h).
//
// The kernels follow cipc_count_active_pt_kernel / cipc_count_active_ee_kernel of consistent_ipc_contact.cu: the
// broad-phase-local pair, its global stencil, pair_d_hat_* and pair_thickness_* (inlined, one line each), the
// distance flag, the flagged squared distance and the activation test, then the doublet and triplet counts of an
// active pair. With `write_pairs` the kernels also store the flag and the active bit of every pair, for the
// correctness check. The EE count kernel also counts the mollified near-parallel pairs as full blocks, which needs
// the rest positions; that count is left out, as it decides no active pair.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <solver/contact_function/distance_flag.h>

namespace
{
constexpr int kBlock = 256;

template <bool WritePairs>
__global__ void classify_pt_kernel(int32_t        n,
                                   const int32_t* pairs,
                                   const int32_t* surf_verts,
                                   const int32_t* faces,
                                   const double*  positions,
                                   const double*  d_hats,
                                   const double*  thicknesses,
                                   int32_t*       flags,
                                   int8_t*        active,
                                   int32_t*       counts)
{
    for(int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x)
    {
        int32_t v0 = surf_verts[pairs[idx * 2]];
        int32_t f  = pairs[idx * 2 + 1];
        int32_t v1 = faces[f * 3 + 0];
        int32_t v2 = faces[f * 3 + 1];
        int32_t v3 = faces[f * 3 + 2];
        double  d_hat = (d_hats[v0] + d_hats[v1]) / 2.0;
        qipc::Vec3 p0(positions[v0 * 3], positions[v0 * 3 + 1], positions[v0 * 3 + 2]);
        qipc::Vec3 p1(positions[v1 * 3], positions[v1 * 3 + 1], positions[v1 * 3 + 2]);
        qipc::Vec3 p2(positions[v2 * 3], positions[v2 * 3 + 1], positions[v2 * 3 + 2]);
        qipc::Vec3 p3(positions[v3 * 3], positions[v3 * 3 + 1], positions[v3 * 3 + 2]);
        uint32_t   flag     = qipc::pt_distance_flag(p0, p1, p2, p3);
        qipc::Vec3 verts[4] = {p0, p1, p2, p3};
        double     d2       = qipc::pt_flagged_distance2(flag, verts);
        double     xi       = thicknesses[v0] + fmax(thicknesses[v1], fmax(thicknesses[v2], thicknesses[v3]));
        double     d_hat_xi = d_hat + xi;
        bool       is_active = !(d2 >= d_hat_xi * d_hat_xi || d2 <= xi * xi);
        if constexpr(WritePairs)
        {
            flags[idx]  = (int32_t)flag;
            active[idx] = is_active ? 1 : 0;
        }
        if(!is_active)
            continue;
        int32_t pc = __popc(flag);
        atomicAdd(counts + 0, (pc == 2) ? 3 : (pc == 3) ? 6 : 10);
        atomicAdd(counts + 1, pc);
    }
}

template <bool WritePairs>
__global__ void classify_ee_kernel(int32_t        n,
                                   const int32_t* pairs,
                                   const int32_t* edges,
                                   const double*  positions,
                                   const double*  d_hats,
                                   const double*  thicknesses,
                                   int32_t*       flags,
                                   int8_t*        active,
                                   int32_t*       counts)
{
    for(int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < n; idx += (int64_t)blockDim.x * gridDim.x)
    {
        int32_t ea = pairs[idx * 2];
        int32_t eb = pairs[idx * 2 + 1];
        int32_t v0 = edges[ea * 2 + 0];
        int32_t v1 = edges[ea * 2 + 1];
        int32_t v2 = edges[eb * 2 + 0];
        int32_t v3 = edges[eb * 2 + 1];
        double  d_hat = (d_hats[v0] + d_hats[v2]) / 2.0;
        qipc::Vec3 p0(positions[v0 * 3], positions[v0 * 3 + 1], positions[v0 * 3 + 2]);
        qipc::Vec3 p1(positions[v1 * 3], positions[v1 * 3 + 1], positions[v1 * 3 + 2]);
        qipc::Vec3 p2(positions[v2 * 3], positions[v2 * 3 + 1], positions[v2 * 3 + 2]);
        qipc::Vec3 p3(positions[v3 * 3], positions[v3 * 3 + 1], positions[v3 * 3 + 2]);
        uint32_t   flag     = qipc::ee_distance_flag(p0, p1, p2, p3);
        qipc::Vec3 verts[4] = {p0, p1, p2, p3};
        double     d2       = qipc::ee_flagged_distance2(flag, verts);
        double     xi = fmax(thicknesses[v0], thicknesses[v1]) + fmax(thicknesses[v2], thicknesses[v3]);
        double     d_hat_xi = d_hat + xi;
        bool       is_active = !(d2 >= d_hat_xi * d_hat_xi || d2 <= xi * xi);
        if constexpr(WritePairs)
        {
            flags[idx]  = (int32_t)flag;
            active[idx] = is_active ? 1 : 0;
        }
        if(!is_active)
            continue;
        int32_t pc = __popc(flag);
        atomicAdd(counts + 0, (pc == 2) ? 3 : (pc == 3) ? 6 : 10);
        atomicAdd(counts + 1, pc);
    }
}

int grid_of(int64_t n)
{
    return (int)std::max<int64_t>(1, std::min<int64_t>((n + kBlock - 1) / kBlock, 1 << 16));
}

void check(const torch::Tensor& t, torch::ScalarType dtype)
{
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == dtype, "CUDA contiguous tensor of the dtype");
}
}  // namespace

// Classify PT and EE pairs; returns (pt flags, pt active, ee flags, ee active, counts) with empty per-pair outputs
// unless write_pairs. counts: [triplets, doublets] of the active pairs, summed over both channels.
std::vector<torch::Tensor> classify(torch::Tensor pt_pairs,
                                    torch::Tensor ee_pairs,
                                    torch::Tensor surf_verts,
                                    torch::Tensor faces,
                                    torch::Tensor edges,
                                    torch::Tensor positions,
                                    torch::Tensor d_hats,
                                    torch::Tensor thicknesses,
                                    bool          write_pairs)
{
    for(auto* t : {&pt_pairs, &ee_pairs, &surf_verts, &faces, &edges})
        check(*t, torch::kInt32);
    for(auto* t : {&positions, &d_hats, &thicknesses})
        check(*t, torch::kFloat64);
    const int32_t n_pt  = (int32_t)(pt_pairs.numel() / 2);
    const int32_t n_ee  = (int32_t)(ee_pairs.numel() / 2);
    auto          i32   = torch::TensorOptions().dtype(torch::kInt32).device(positions.device());
    auto          i8    = torch::TensorOptions().dtype(torch::kInt8).device(positions.device());
    auto          pt_fl = torch::empty({write_pairs ? n_pt : 0}, i32);
    auto          pt_ac = torch::empty({write_pairs ? n_pt : 0}, i8);
    auto          ee_fl = torch::empty({write_pairs ? n_ee : 0}, i32);
    auto          ee_ac = torch::empty({write_pairs ? n_ee : 0}, i8);
    auto          counts = torch::zeros({2}, i32);
    auto          stream = at::cuda::getCurrentCUDAStream();
    if(n_pt > 0)
    {
        auto kernel = write_pairs ? classify_pt_kernel<true> : classify_pt_kernel<false>;
        kernel<<<grid_of(n_pt), kBlock, 0, stream>>>(n_pt,
                                                     pt_pairs.data_ptr<int32_t>(),
                                                     surf_verts.data_ptr<int32_t>(),
                                                     faces.data_ptr<int32_t>(),
                                                     positions.data_ptr<double>(),
                                                     d_hats.data_ptr<double>(),
                                                     thicknesses.data_ptr<double>(),
                                                     pt_fl.data_ptr<int32_t>(),
                                                     pt_ac.data_ptr<int8_t>(),
                                                     counts.data_ptr<int32_t>());
    }
    if(n_ee > 0)
    {
        auto kernel = write_pairs ? classify_ee_kernel<true> : classify_ee_kernel<false>;
        kernel<<<grid_of(n_ee), kBlock, 0, stream>>>(n_ee,
                                                     ee_pairs.data_ptr<int32_t>(),
                                                     edges.data_ptr<int32_t>(),
                                                     positions.data_ptr<double>(),
                                                     d_hats.data_ptr<double>(),
                                                     thicknesses.data_ptr<double>(),
                                                     ee_fl.data_ptr<int32_t>(),
                                                     ee_ac.data_ptr<int8_t>(),
                                                     counts.data_ptr<int32_t>());
    }
    return {pt_fl, pt_ac, ee_fl, ee_ac, counts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("classify", &classify, "CGQ classification-only narrow phase of PT and EE pairs");
}
