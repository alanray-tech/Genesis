# CGQ side of the benchmark, with the Python of a built cuda-graph-qipc checkout.
#   powershell -File bench\ipc_bvh\run_cgq.ps1 -Stage capture [-Scenes kimono,...]
#   powershell -File bench\ipc_bvh\run_cgq.ps1 -Stage time [-Scenes kimono,...]
# capture: frames.npz and gt.npz per scene (cgq_side.py capture).
# time: per scene, the recompute() loop without tracing (wall times), then under nsys (kernel timeline, parsed by
# cgq_nsys.py into results/<scene>_cgq.json), then the narrow-phase extension on CGQ's and on the bvh.py candidates
# (cgq_narrow.py, which needs the candidates run_upstream.ps1 saved).
[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet('capture', 'time')][string]$Stage = 'time',
    [string[]]$Scenes = @('two_bunnies', 'cloth_stack', 'wrecking_ball', 'kimono', 'cloth_stack_x3'),
    [string]$Cgq = 'C:\Users\81946\Projects\CudaGraphQIPC\cuda-graph-qipc-bvh-bench',
    [int]$Repeats = 20
)
$ErrorActionPreference = 'Stop'
$py = "$Cgq\.venv\Scripts\python.exe"
$nsysDir = "$PSScriptRoot\data\nsys"
New-Item -ItemType Directory -Force $nsysDir | Out-Null
foreach ($scene in $Scenes) {
    if ($Stage -eq 'capture') {
        & $py "$PSScriptRoot\cgq_side.py" --cgq $Cgq --scene $scene capture
        if ($LASTEXITCODE) { exit $LASTEXITCODE }
        continue
    }
    & $py "$PSScriptRoot\cgq_side.py" --cgq $Cgq --scene $scene --repeats $Repeats time
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    $trace = "$nsysDir\cgq_$scene"
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node -t cuda,nvtx `
        --force-overwrite true -o $trace $py "$PSScriptRoot\cgq_side.py" --cgq $Cgq --scene $scene `
        --repeats $Repeats --traced time
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    nsys export --type sqlite --force-overwrite true -o "$trace.sqlite" "$trace.nsys-rep"
    & $py "$PSScriptRoot\cgq_nsys.py" --scene $scene "$trace.sqlite"
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    & $py "$PSScriptRoot\cgq_narrow.py" --cgq $Cgq --scene $scene --repeats $Repeats
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}
