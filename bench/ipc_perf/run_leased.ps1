# Every timed run of the profile, to run in one exclusive GPU lease: wall times, nsys windows and cProfile, on both sides.
#   powershell -File bench\ipc_perf\run_leased.ps1 [-Tree <genesis tree>] [-Cgq <cuda-graph-qipc checkout>]
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Tree = 'C:\Users\81946\Projects\GenesisSimEngine\genesis-world-ipc-perf',
    [string]$Cgq = 'C:\Users\81946\Projects\CudaGraphQIPC\cuda-graph-qipc-p2a-aeb58956',
    [string[]]$Scenes = @('cloth_on_box', 'crossed_drop')
)
$results = "$PSScriptRoot\results"
New-Item -ItemType Directory -Force $results | Out-Null
$cgqPy = "$Cgq\.venv\Scripts\python.exe"
# The nsys windows: frames in contact on both sides
$windows = @{ cloth_on_box = @(40, 60); crossed_drop = @(60, 90) }
foreach ($scene in $Scenes) {
    $start, $end = $windows[$scene]
    foreach ($tol in '1e-6', '0.05') {
        powershell -File "$PSScriptRoot\run.ps1" -Tree $Tree -Script genesis_profile.py --scene $scene --mode wall `
            --velocity-tol $tol
        & $cgqPy "$PSScriptRoot\cgq_profile.py" --cgq $Cgq --scene $scene --mode wall --velocity-tol $tol
    }
    powershell -File "$PSScriptRoot\run.ps1" -Tree $Tree -Nsys "$results\genesis_$scene" -Script genesis_profile.py `
        --scene $scene --mode nsys --start $start --end $end
    & nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop `
        --force-overwrite=true -o "$results\cgq_$scene" $cgqPy "$PSScriptRoot\cgq_profile.py" --cgq $Cgq --scene $scene `
        --mode nsys --start $start --end $end
    powershell -File "$PSScriptRoot\run.ps1" -Tree $Tree -Script genesis_profile.py --scene $scene --mode cprofile `
        --start $start --end $end
}
