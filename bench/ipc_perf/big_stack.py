"""Run a Python script in a thread with a 128 MiB stack (the largest Python accepts on Windows), its arguments passed on.

Under Nsight Compute, the injected profiler leaves the main thread too little stack for the recursion of the Quadrants
frontend, which then dies of a stack overflow while compiling the first kernels.

    python big_stack.py genesis_profile.py --scene crossed_drop --mode nsys --start 70 --end 71
"""

import runpy
import sys
import threading

script, *script_args = sys.argv[1:]
sys.argv = [script, *script_args]
threading.stack_size(1 << 27)
errors = []


def run() -> None:
    try:
        runpy.run_path(script, run_name="__main__")
    except BaseException as exception:
        errors.append(exception)
        raise


thread = threading.Thread(target=run)
thread.start()
thread.join()
sys.exit(1 if errors else 0)
