"""Reference side of the IPC formula parity: CGQ's contact functions on the input rows, compiled as a CUDA extension.

    <cgq>/.venv/Scripts/python.exe bench/ipc_formulas/cgq_side.py --cgq <cgq> [--fmad true|false]

'<cgq>' is a checkout of the ground truth commit (cases.GROUND_TRUTH), or of a fix on top of it given a '--tag' (its
build folder and output file take the tag as suffix, '--cases' selecting the cases it changes). Reads <data>/inputs.npz
(inputs.py) and writes <data>/out_cgq_fmad-<fmad><tag>.npz, <data> being cases.DATA, one array per case and input set,
named '<case>/<set>'.
'--fmad' is nvcc's contraction of multiplies and adds into fused multiply-adds: true is nvcc's default and CGQ's
production build, false evaluates every operation with its own rounding.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from cases import CANDIDATE_REACH_SCALE, CASES, DATA, GROUND_TRUTH, n_outputs

HERE = Path(__file__).resolve().parent


def load_extension(cgq: Path, is_fmad: bool, tag: str):
    from torch.utils.cpp_extension import load

    sys.path.insert(0, str(cgq))
    from build import _msvc_env

    os.environ.update(_msvc_env())
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "{}.{}".format(*torch.cuda.get_device_capability())
    native = cgq / "qipc" / "_src" / "native"
    external = cgq / "external"
    build_dir = DATA / f"cgq_build_fmad-{str(is_fmad).lower()}{tag}"
    build_dir.mkdir(parents=True, exist_ok=True)
    # torch waits without end on the lock of a build that was killed, so a leftover lock is an error to clear by hand
    if (build_dir / "lock").exists():
        raise SystemExit(f"{build_dir / 'lock'} exists: another build runs there, or a killed one left it")
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
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--tag", default="", help="suffix of the build folder and of the output file")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.cgq, text=True)
    is_on_top = subprocess.run(["git", "merge-base", "--is-ancestor", GROUND_TRUTH, "HEAD"], cwd=args.cgq).returncode == 0
    if not (head.startswith(GROUND_TRUTH) or (args.tag and is_on_top)):
        raise SystemExit(f"{args.cgq} is at {head.strip()}: the ground truth {GROUND_TRUTH}, or a fix on top with '--tag'")
    ext = load_extension(args.cgq.resolve(), args.fmad == "true", args.tag)
    print(f"[cgq] extension loaded, {len(ext.case_names())} cases", flush=True)
    if args.build_only:
        return

    ext.set_candidate_context(0.0, CANDIDATE_REACH_SCALE)
    outputs = {}
    with np.load(DATA / "inputs.npz") as inputs:
        for key in inputs.files:
            case = CASES[key.split("/")[0]]
            if args.cases and case.name not in args.cases:
                continue
            rows = torch.as_tensor(inputs[key], dtype=torch.float64, device="cuda").contiguous()
            out = torch.full((rows.shape[0], n_outputs(case)), np.nan, dtype=torch.float64, device="cuda")
            ext.run(case.name, rows, out)
            torch.cuda.synchronize()
            outputs[key] = out.cpu().numpy()
            print(f"[cgq] {key}: {rows.shape[0]} rows", flush=True)
    path = DATA / f"out_cgq_fmad-{args.fmad}{args.tag}.npz"
    np.savez(path, **outputs)
    print(f"[cgq] wrote {path}")


if __name__ == "__main__":
    main()
