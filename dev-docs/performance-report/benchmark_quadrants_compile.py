"""Phase-separated Quadrants compile and cached-restart benchmark.

Run this script in a new Python process for every sample. Population and restart
runs must use the same explicit cache directory; cold and population runs refuse
non-empty directories so they cannot accidentally consume an old artifact.
"""

from __future__ import annotations

import argparse
import cProfile
import ctypes
import dataclasses
import functools
import hashlib
import importlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROCESS_START = time.perf_counter()
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "examples" / "newton_coupling"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-mode",
        choices=("cold", "populate", "restart", "offline-only-populate", "offline-only-restart"),
        required=True,
    )
    parser.add_argument(
        "--compiler-mode",
        choices=("production", "development-o3", "development-o1"),
        default="development-o3",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ee-query", choices=("dual", "warp"), default="dual")
    parser.add_argument("--compile-threads", type=int, default=4)
    parser.add_argument(
        "--guard-steps",
        type=int,
        default=0,
        help="Synchronized post-compile steps used to guard steady-state runtime",
    )
    parser.add_argument(
        "--ast-profile-dir",
        type=Path,
        default=None,
        help="Optional cProfile output directory for both SimEngine._step_kernel AST passes",
    )
    return parser.parse_args()


def _prepare_cache(args: argparse.Namespace) -> tuple[bool, bool]:
    cache_dir = args.cache_dir.resolve()
    has_entries = cache_dir.exists() and any(cache_dir.iterdir())
    is_population = args.cache_mode in ("cold", "populate", "offline-only-populate")
    if is_population and has_entries:
        raise RuntimeError(f"{args.cache_mode} requires an empty cache directory: {cache_dir}")
    if not is_population and not has_entries:
        raise RuntimeError(f"{args.cache_mode} requires a populated cache directory: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    offline_cache = args.cache_mode != "cold"
    src_ll_cache = args.cache_mode in ("populate", "restart")
    os.environ["QD_OFFLINE_CACHE_FILE_PATH"] = str(cache_dir)
    os.environ["QD_OFFLINE_CACHE"] = "1" if offline_cache else "0"
    os.environ["QD_OFFLINE_CACHE_CLEANING_POLICY"] = "never"
    os.environ["QD_NUM_THREADS"] = str(args.compile_threads)
    return offline_cache, src_ll_cache


def _git_info(repo: Path) -> dict[str, Any]:
    def run(*command: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(repo), *command),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    status = run("status", "--short")
    patch = run("diff", "--binary", "--no-ext-diff")
    return {
        "path": str(repo),
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status": status.splitlines(),
        "tracked_patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest() if patch else None,
    }


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return repr(value)


def _public_scalar_attributes(obj: object) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception as exc:  # pragma: no cover - diagnostic collection only
            result[name] = f"<error: {exc}>"
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value
    return result


def _install_python_timers(ast_profile_dir: Path | None) -> list[dict[str, Any]]:
    from quadrants.lang import impl
    from quadrants.lang._fast_caching import function_hasher, src_hasher
    from quadrants.lang._fast_caching.python_side_cache import PythonSideCache

    kernel_module = importlib.import_module("quadrants.lang.kernel")
    events: list[dict[str, Any]] = []

    def kernel_name(obj: object) -> str:
        func = getattr(obj, "func", None)
        return getattr(func, "__qualname__", getattr(func, "__name__", type(obj).__name__))

    def patch_method(cls: type, name: str, stage: str) -> None:
        original = getattr(cls, name)

        @functools.wraps(original)
        def timed(self, *args, **kwargs):
            start = time.perf_counter()
            event = {
                "stage": stage,
                "kernel": kernel_name(self),
            }
            if name == "materialize" and args:
                key = args[0]
                event["already_materialized"] = key in getattr(self, "materialized_kernels", {})
            if name == "launch_kernel" and len(args) >= 3:
                event["compiled_data_available"] = args[2] is not None
            try:
                return original(self, *args, **kwargs)
            finally:
                event["elapsed_s"] = time.perf_counter() - start
                if name == "launch_kernel" and getattr(self, "use_graph", False):
                    event["offloaded_tasks"] = impl.get_runtime().prog.get_num_offloaded_tasks_on_last_call()
                observations = getattr(self, "src_ll_cache_observations", None)
                if observations is not None:
                    event["src_ll_cache"] = _json_value(observations)
                events.append(event)

        setattr(cls, name, timed)

    patch_method(kernel_module.Kernel, "_try_load_fastcache", "fastcache_lookup_total")
    patch_method(kernel_module.Kernel, "materialize", "kernel_materialize")
    patch_method(kernel_module.Kernel, "launch_kernel", "kernel_launch")

    original_ast_call = kernel_module.ASTGenerator.__call__

    @functools.wraps(original_ast_call)
    def timed_ast_call(self, *args, **kwargs):
        name = kernel_name(self.current_kernel)
        pass_index = getattr(self.ctx.global_context, "pass_idx", None)
        profile = None
        if ast_profile_dir is not None and name == "SimEngine._step_kernel":
            ast_profile_dir.mkdir(parents=True, exist_ok=True)
            profile = cProfile.Profile()
            profile.enable()
        start = time.perf_counter()
        try:
            return original_ast_call(self, *args, **kwargs)
        finally:
            if profile is not None:
                profile.disable()
                profile.dump_stats(ast_profile_dir / f"step-kernel-pass-{pass_index}.prof")
            events.append(
                {
                    "stage": "ast_transform",
                    "kernel": name,
                    "pass": pass_index,
                    "only_parse_function_def": self.only_parse_function_def,
                    "elapsed_s": time.perf_counter() - start,
                }
            )

    kernel_module.ASTGenerator.__call__ = timed_ast_call

    original_read_file = function_hasher._read_file

    @functools.wraps(original_read_file)
    def timed_read_file(function_info):
        start = time.perf_counter()
        try:
            lines = original_read_file(function_info)
            return lines
        finally:
            events.append(
                {
                    "stage": "source_file_read",
                    "path": function_info.filepath,
                    "start_lineno": function_info.start_lineno,
                    "end_lineno": function_info.end_lineno,
                    "elapsed_s": time.perf_counter() - start,
                }
            )

    function_hasher._read_file = timed_read_file

    original_cache_load = PythonSideCache.try_load

    @functools.wraps(original_cache_load)
    def timed_cache_load(self, fast_cache_key: str):
        start = time.perf_counter()
        result = None
        try:
            result = original_cache_load(self, fast_cache_key)
            return result
        finally:
            events.append(
                {
                    "stage": "python_cache_file_load",
                    "key": fast_cache_key,
                    "hit": result is not None,
                    "elapsed_s": time.perf_counter() - start,
                }
            )

    PythonSideCache.try_load = timed_cache_load

    original_touch = PythonSideCache._touch

    @functools.wraps(original_touch)
    def timed_touch(self, filepath):
        start = time.perf_counter()
        try:
            return original_touch(self, filepath)
        finally:
            events.append(
                {
                    "stage": "python_cache_touch",
                    "path": filepath,
                    "elapsed_s": time.perf_counter() - start,
                }
            )

    PythonSideCache._touch = timed_touch

    def patch_function(module: object, name: str, stage: str) -> None:
        original: Callable[..., Any] = getattr(module, name)

        @functools.wraps(original)
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                events.append({"stage": stage, "elapsed_s": time.perf_counter() - start})

        setattr(module, name, timed)

    patch_function(src_hasher, "make_source_config_key", "source_config_hash")
    patch_function(src_hasher, "load_pruning_info", "fastcache_l1_load_validate")
    patch_function(src_hasher, "compute_narrow_args_hash", "fastcache_narrow_args_hash")
    patch_function(src_hasher, "load", "fastcache_l2_load_validate")
    return events


def _kernel_cache_observations(engine: object) -> dict[str, Any]:
    observations: dict[str, Any] = {}
    for name in ("_initialize_global_resources", "_init_contact_kernel", "_step_kernel"):
        callable_obj = getattr(engine, name)
        primal = getattr(callable_obj, "_primal", None)
        if primal is None:
            observations[name] = {"error": "kernel primal unavailable"}
            continue
        observations[name] = {
            "source": _json_value(primal.src_ll_cache_observations),
            "frontend": _json_value(primal.fe_ll_cache_observations),
            "launch": _json_value(primal.launch_observations),
        }
    return observations


def _telemetry(engine: object, qd_to_numpy: Callable[[Any], Any]) -> dict[str, Any]:
    contact = engine.contact
    dual = contact.broad_phase.ee_dual_state
    return {
        "newton": engine.get_newton_iters(),
        "max_pcg": engine.get_max_pcg_iters(),
        "total_pcg": engine.get_total_pcg_iters(),
        "line_search": engine.get_max_ls_iters(),
        "pairs": {
            "pt": int(qd_to_numpy(contact.n_pairs_pt)),
            "ee": int(qd_to_numpy(contact.n_pairs_ee)),
            "ph": int(qd_to_numpy(contact.n_pairs_ph)),
            "active": int(qd_to_numpy(contact.n_active_pairs)),
        },
        "ccd_alpha": float(qd_to_numpy(contact.frame_ccd_alpha)),
        "overflow": {
            "pair": int(qd_to_numpy(contact.overflow_flag)),
            "assembly": int(qd_to_numpy(contact.count_overflow_flag)),
            "padding": int(qd_to_numpy(contact.contact_padding_overflow)),
            "triplet": int(qd_to_numpy(engine.global_linear_system.triplet_overflow)),
            "friction": int(qd_to_numpy(contact.friction_overflow_flag)),
            "edge_triangle": int(qd_to_numpy(contact.et_overflow_flag)),
            "dual": int(qd_to_numpy(dual.overflow_bits)),
        },
    }


def _summarize_python_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, list[float]] = {}
    source_paths: dict[str, dict[str, float | int]] = {}
    for event in events:
        stage = str(event["stage"])
        by_stage.setdefault(stage, []).append(float(event["elapsed_s"]))
        if stage == "source_file_read":
            path = str(event["path"])
            entry = source_paths.setdefault(path, {"count": 0, "elapsed_s": 0.0})
            entry["count"] = int(entry["count"]) + 1
            entry["elapsed_s"] = float(entry["elapsed_s"]) + float(event["elapsed_s"])
    return {
        "by_stage": {
            stage: {
                "count": len(samples),
                "total_s": sum(samples),
                "max_s": max(samples),
            }
            for stage, samples in sorted(by_stage.items())
        },
        "source_reads_by_path": source_paths,
    }


def main() -> None:
    args = _parse_args()
    offline_cache, src_ll_cache = _prepare_cache(args)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

    timings: dict[str, float] = {}

    start = time.perf_counter()
    import quadrants as qd
    from quadrants._lib import core as qd_core
    from quadrants.lang import impl as qd_impl

    timings["import_quadrants_s"] = time.perf_counter() - start

    start = time.perf_counter()
    import genesis as gs
    import torch

    timings["import_genesis_s"] = time.perf_counter() - start

    sys.path.insert(0, str(EXAMPLE_DIR))
    start = time.perf_counter()
    from franka_cloth_cube_teleop import (
        HOME_QPOS,
        build_franka_cloth_cube_engine,
        build_franka_cloth_cube_scene,
    )
    from genesis.utils.misc import qd_to_numpy

    timings["import_workload_s"] = time.perf_counter() - start

    python_events = _install_python_timers(args.ast_profile_dir)

    start = time.perf_counter()
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    timings["gs_init_s"] = time.perf_counter() - start

    compiler_modes = {
        "production": (True, 3),
        "development-o3": (False, 3),
        "development-o1": (False, 1),
    }
    advanced_optimization, external_optimization_level = compiler_modes[args.compiler_mode]
    qd.cfg.advanced_optimization = advanced_optimization
    qd.cfg.external_optimization_level = external_optimization_level
    qd_impl.get_runtime().src_ll_cache = src_ll_cache

    start = time.perf_counter()
    scene, franka, arm_dofs, finger_dofs = build_franka_cloth_cube_scene(show_viewer=False)
    timings["scene_author_build_s"] = time.perf_counter() - start

    qd.profiler.clear_scoped_profiler_info()
    start = time.perf_counter()
    engine = build_franka_cloth_cube_engine(scene, ee_query=args.ee_query)
    qd.sync()
    timings["engine_build_init_s"] = time.perf_counter() - start

    end_effector = franka.get_link("hand")
    target_pos = end_effector.get_pos().cpu().numpy().reshape(3)
    target_quat = end_effector.get_quat().cpu().numpy().reshape(4)
    start = time.perf_counter()
    target_qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=target_pos,
        quat=target_quat,
        init_qpos=franka.get_qpos(),
        max_samples=1,
        max_solver_iters=8,
        damping=0.05,
        max_step_size=0.1,
        dofs_idx_local=arm_dofs,
    )
    franka.control_dofs_position(target_qpos[arm_dofs], dofs_idx_local=arm_dofs)
    franka.control_dofs_position(0.04, dofs_idx_local=finger_dofs)
    timings["headless_control_s"] = time.perf_counter() - start

    start = time.perf_counter()
    engine.step()
    qd.sync()
    timings["first_step_compile_launch_sync_s"] = time.perf_counter() - start
    timings["process_to_first_completed_step_s"] = time.perf_counter() - PROCESS_START

    first_telemetry = _telemetry(engine, qd_to_numpy)
    graph_task_counts = {
        event["kernel"]: event["offloaded_tasks"]
        for event in python_events
        if event["stage"] == "kernel_launch" and "offloaded_tasks" in event
    }

    guard_ms: list[float] = []
    guard_telemetry: list[dict[str, Any]] = []
    for _ in range(args.guard_steps):
        start = time.perf_counter()
        engine.step()
        qd.sync()
        guard_ms.append((time.perf_counter() - start) * 1000.0)
        guard_telemetry.append(_telemetry(engine, qd_to_numpy))

    guard_summary = None
    if guard_ms:
        ordered = sorted(guard_ms)
        p95_index = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
        guard_summary = {
            "samples_ms": guard_ms,
            "median_ms": statistics.median(guard_ms),
            "mean_ms": statistics.fmean(guard_ms),
            "p95_ms": ordered[p95_index],
            "maxima": {
                "newton": max(item["newton"] for item in guard_telemetry),
                "max_pcg": max(item["max_pcg"] for item in guard_telemetry),
                "total_pcg": max(item["total_pcg"] for item in guard_telemetry),
                "line_search": max(item["line_search"] for item in guard_telemetry),
                "pairs_ee": max(item["pairs"]["ee"] for item in guard_telemetry),
            },
            "min_ccd_alpha": min(item["ccd_alpha"] for item in guard_telemetry),
            "overflow_or": {
                name: int(any(item["overflow"][name] for item in guard_telemetry))
                for name in guard_telemetry[0]["overflow"]
            },
        }

    qd_repo = Path(qd.__file__).resolve().parents[2]
    try:
        driver_version = subprocess.run(
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        driver_version = None

    report = {
        "schema_version": 1,
        "benchmark": {
            "cache_mode": args.cache_mode,
            "compiler_mode": args.compiler_mode,
            "cache_dir": str(args.cache_dir.resolve()),
            "offline_cache": offline_cache,
            "src_ll_cache": src_ll_cache,
            "nvidia_driver_cache_disabled": os.environ.get("CUDA_CACHE_DISABLE") == "1",
            "compile_threads": args.compile_threads,
            "ee_query": args.ee_query,
            "guard_steps": args.guard_steps,
            "ast_profile_dir": str(args.ast_profile_dir.resolve()) if args.ast_profile_dir is not None else None,
            "python_timing_instrumentation": True,
        },
        "revisions": {
            "quadrants": _git_info(qd_repo),
            "genesis": _git_info(REPO_ROOT),
            "quadrants_python_path": str(Path(qd.__file__).resolve()),
            "quadrants_native_path": str(Path(qd_core.__file__).resolve()),
            "quadrants_native_commit": qd_core.get_commit_hash(),
            "quadrants_version": qd.__version_str__,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "llvm": qd_core.get_llvm_target_support(),
            "gpu": torch.cuda.get_device_name(gs.device) if gs.device.type == "cuda" else str(gs.device),
            "driver": driver_version,
            "qd_environment": {
                name: value
                for name, value in os.environ.items()
                if name.startswith("QD_") or name == "CUDA_CACHE_DISABLE"
            },
        },
        "compiler_config": _public_scalar_attributes(qd.cfg),
        "timings": timings,
        "python_timing_summary": _summarize_python_events(python_events),
        "python_timing_events": python_events,
        "fastcache_observations": _kernel_cache_observations(engine),
        "graph_task_counts": graph_task_counts,
        "first_step_telemetry": first_telemetry,
        "runtime_guard": guard_summary,
        "scoped_profiler": {
            "format": "stdout",
            "begin_marker": "BEGIN_QUADRANTS_SCOPED_PROFILER",
            "end_marker": "END_QUADRANTS_SCOPED_PROFILER",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("BEGIN_QUADRANTS_SCOPED_PROFILER", flush=True)
    qd.profiler.print_scoped_profiler_info()
    if sys.platform == "win32":
        ctypes.CDLL("ucrtbase").fflush(None)
    print("END_QUADRANTS_SCOPED_PROFILER", flush=True)
    print(json.dumps({"output": str(args.output.resolve()), "timings": timings}, indent=2), flush=True)


if __name__ == "__main__":
    main()
