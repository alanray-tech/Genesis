"""Static resource usage and instruction mix of GPU kernels, from Quadrants PTX (Genesis) or a CUDA binary (CGQ).

PTX files are compiled with ptxas for sm_120, whose verbose output gives the registers, the stack frame and the spill
bytes of every entry; their SASS (cuobjdump -sass) gives the static instruction mix of every function: instructions,
fp64 instructions (D* opcodes and MUFU.*64), local loads and stores (LDL / STL, the spills and dynamically indexed local
arrays), global atomics and reductions, branches. A binary (CGQ's native module) gives the same through
cuobjdump -res-usage and -sass. Functions are kept when their name matches the regex.

    python sass_stats.py --ptx <file.ptx> [...] --match "kernel_assemble_rows"
    python sass_stats.py --binary <qipc native .pyd> --match "cipc_filter_(assemble|energy)_ee"
"""

import argparse
import collections
import json
import re
import subprocess
import tempfile
from pathlib import Path

CUDA = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")

parser = argparse.ArgumentParser()
parser.add_argument("--ptx", nargs="*", default=[])
parser.add_argument("--binary", default=None)
parser.add_argument("--match", default=".")
parser.add_argument("--out", default=None)
args = parser.parse_args()
pattern = re.compile(args.match)


def sass_mix(sass: str) -> dict:
    """The instruction mix of every function of a cuobjdump -sass listing."""
    mix = collections.defaultdict(collections.Counter)
    function = None
    for line in sass.splitlines():
        header = re.match(r"\s*Function : (\S+)", line)
        if header:
            function = header.group(1)
            continue
        instruction = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if function is None or instruction is None:
            continue
        opcode = instruction.group(1)
        base = opcode.split(".")[0]
        counts = mix[function]
        counts["instructions"] += 1
        if base in ("DFMA", "DMUL", "DADD", "DSETP", "DMNMX", "DSET") or (base == "MUFU" and "64" in opcode):
            counts["fp64"] += 1
        if base == "LDL":
            counts["local_loads"] += 1
        if base == "STL":
            counts["local_stores"] += 1
        if base in ("ATOM", "ATOMG", "RED", "REDG"):
            counts["global_atomics"] += 1
        if base in ("BRA", "BRX", "JMP", "JMX"):
            counts["branches"] += 1
    return mix


rows = {}
if args.binary:
    resources = subprocess.run(
        [str(CUDA / "cuobjdump.exe"), "-res-usage", args.binary], capture_output=True, text=True
    ).stdout
    function = None
    for line in resources.splitlines():
        header = re.match(r"\s*Function (\S+):", line)
        if header:
            function = header.group(1)
            continue
        usage = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", line)
        if function and usage and pattern.search(function):
            sass = subprocess.run(
                [str(CUDA / "cuobjdump.exe"), "-sass", "-fun", function, args.binary], capture_output=True, text=True
            ).stdout
            rows[function] = {
                "regs": int(usage.group(1)),
                "stack": int(usage.group(2)),
                "shared": int(usage.group(3)),
                "local": int(usage.group(4)),
                **sass_mix(sass).get(function, {}),
            }
for ptx in args.ptx:
    with tempfile.TemporaryDirectory() as tmp:
        cubin = Path(tmp) / "kernel.cubin"
        compiled = subprocess.run(
            [str(CUDA / "ptxas.exe"), "-arch=sm_120", "-v", "-o", str(cubin), ptx], capture_output=True, text=True
        )
        verbose = compiled.stdout + compiled.stderr
        sass = subprocess.run([str(CUDA / "cuobjdump.exe"), "-sass", str(cubin)], capture_output=True, text=True).stdout
    mix = sass_mix(sass)
    function = None
    for line in verbose.splitlines():
        header = re.search(r"Compiling entry function '(\S+)'", line)
        if header:
            function = header.group(1)
            continue
        frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        registers = re.search(r"Used (\d+) registers", line)
        if function is None or not pattern.search(function):
            continue
        entry = rows.setdefault(f"{Path(ptx).stem[:10]}:{function}", {})
        if frame:
            entry.update(stack=int(frame.group(1)), spill_stores=int(frame.group(2)), spill_loads=int(frame.group(3)))
        if registers:
            entry.update(regs=int(registers.group(1)), **mix.get(function, {}))

if args.out:
    Path(args.out).write_text(json.dumps(rows, indent=1))
keys = ("regs", "stack", "spill_stores", "spill_loads", "local", "instructions", "fp64", "local_loads", "local_stores",
        "global_atomics", "branches")
print("function | " + " | ".join(keys))
for name, entry in rows.items():
    print(f"{name[:90]} | " + " | ".join(str(entry.get(key, "")) for key in keys))
