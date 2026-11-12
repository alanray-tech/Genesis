"""Launch one tiny Quadrants kernel on the GPU, then one Genesis kernel through gs.init, to check that the process
survives Nsight Compute's injection (run_ncu.ps1 crashed in the frontend).

    python ncu_smoke.py [--genesis]
"""

import argparse

import quadrants as qd

parser = argparse.ArgumentParser()
parser.add_argument("--genesis", action="store_true")
args = parser.parse_args()

if args.genesis:
    import genesis as gs

    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.build()
    scene.step()
    print("genesis step ok", flush=True)
else:
    qd.init(arch=qd.cuda)
    values = qd.field(qd.f64, shape=1024)

    @qd.kernel
    def kernel_fill(values: qd.template()):
        for i in values:
            values[i] = i * 0.5

    kernel_fill(values)
    qd.sync()
    print("quadrants kernel ok", flush=True)
