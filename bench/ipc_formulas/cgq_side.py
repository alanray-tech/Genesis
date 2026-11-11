"""Reference side of the IPC formula parity: CGQ's contact functions on the input rows, compiled as a CUDA extension.

    <cgq>/.venv/Scripts/python.exe bench/ipc_formulas/cgq_side.py --cgq <cgq> [--fmad true|false]

Reads data/inputs.npz (inputs.py) and writes data/out_cgq_fmad-<fmad>.npz, one array per case and input set, named
'<case>/<set>'. '--fmad' is nvcc's contraction of multiplies and adds into fused multiply-adds: true is nvcc's default
and CGQ's production build, false evaluates every operation with its own rounding.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from cases import CANDIDATE_REACH_SCALE, CASES, n_outputs

HERE = Path(__file__).resolve().parent


def load_extension(cgq: Path, is_fmad: bool):
    from torch.utils.cpp_extension import load

    sys.path.insert(0, str(cgq))
    from build import _msvc_env

    os.environ.update(_msvc_env())
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "{}.{}".format(*torch.cuda.get_device_capability()))
    native = cgq / "qipc" / "_src" / "native"
    external = cgq / "external"
    build_dir = HERE / "data" / f"cgq_build_fmad-{str(is_fmad).lower()}"
    build_dir.mkdir(parents=True, exist_ok=True)
    defines = ["-DNOMINMAX", "-DQIPC_RUNTIME_CHECK=0", "-DFMT_HEADER_ONLY", "-DFMT_UNICODE=0", "-DSPDLOG_HEADER_ONLY", "-DSPDLOG_FMT_EXTERNAL"]
    return load(
        name=f"cgq_formulas_fmad_{str(is_fmad).lower()}",
        sources=[str(HERE / "cgq_formulas" / "cgq_formulas.cu")],
        extra_include_paths=[
            str(native),
            str(external / "eigen"),
            str(external / "fmt" / "include"),
            str(external / "spdlog" / "include"),
            str(external / "cpptrace" / "include"),
        ],
        extra_cuda_cflags=[
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "--extended-lambda",
            "-O3",
            f"--fmad={str(is_fmad).lower()}",
            "-Xcompiler",
            "/utf-8",
            *defines,
        ],
        extra_cflags=["/std:c++20", "/utf-8", *[d.replace("-D", "/D") for d in defines]]
        if os.name == "nt"
        else ["-std=c++20"],
        build_directory=str(build_dir),
        verbose=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cgq", type=Path, required=True)
    parser.add_argument("--fmad", choices=("true", "false"), default="true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    ext = load_extension(args.cgq.resolve(), args.fmad == "true")
    print(f"[cgq] extension loaded, {len(ext.case_names())} cases", flush=True)
    if args.build_only:
        return

    ext.set_candidate_context(0.0, CANDIDATE_REACH_SCALE)
    outputs = {}
    with np.load(HERE / "data" / "inputs.npz") as inputs:
        for key in inputs.files:
            case = CASES[key.split("/")[0]]
            rows = torch.as_tensor(inputs[key], dtype=torch.float64, device="cuda").contiguous()
            out = torch.full((rows.shape[0], n_outputs(case)), np.nan, dtype=torch.float64, device="cuda")
            ext.run(case.name, rows, out)
            torch.cuda.synchronize()
            outputs[key] = out.cpu().numpy()
            print(f"[cgq] {key}: {rows.shape[0]} rows", flush=True)
    path = HERE / "data" / f"out_cgq_fmad-{args.fmad}.npz"
    np.savez(path, **outputs)
    print(f"[cgq] wrote {path}")


if __name__ == "__main__":
    main()
