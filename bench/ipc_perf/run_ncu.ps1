# Profile the kernels of one frame of a scene with Nsight Compute (full set), Genesis or CGQ side.
#   powershell -File bench\ipc_perf\run_ncu.ps1 -Side genesis|cgq -Scene crossed_drop -Frame 70 -Kernels '<regex>' -Count 400 -Tag main
# The frame is selected by its NVTX range 'step <frame>' (the nsys mode of the profiling scripts), the kernels by a regex
# on their function names; at most 'Count' launches are profiled, in launch order.
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Side,
    [string]$Scene,
    [int]$Frame,
    [string]$Kernels,
    [int]$Count = 400,
    [string]$Tag = 'main',
    [string]$Tree = 'C:\Users\81946\Projects\GenesisSimEngine\genesis-world-ipc-perf',
    [string]$Overlay = 'C:\Users\81946\Projects\GenesisSimEngine\scratch\pr19\qd-devfast-dbg-overlay',
    [string]$Cgq = 'C:\Users\81946\Projects\CudaGraphQIPC\cuda-graph-qipc-p2a-aeb58956',
    [string]$Extra = ''
)
$extraArgs = @($Extra -split ' ' | Where-Object { $_ })
# ncu.bat goes through cmd, which splits a regex at '|': call the executable
$ncu = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.3.1\target\windows-desktop-win7-x64\ncu.exe'
$results = "$PSScriptRoot\results"
$out = "$results\ncu_${Side}_${Scene}_f${Frame}_$Tag"
$scriptArgs = @('--scene', $Scene, '--mode', 'nsys', '--start', $Frame, '--end', ($Frame + 1))
if ($Side -eq 'genesis') {
    $env:PYTHONPATH = "$Overlay;$Tree"
    $cache = 'C:\Users\81946\Projects\GenesisSimEngine\scratch\pr19\cp-run\cache-ipc-perf'
    $env:QD_OFFLINE_CACHE_FILE_PATH = "$cache\quadrants"
    $env:GS_CACHE_FILE_PATH = "$cache\genesis"
    $py = 'C:\Users\81946\Projects\GenesisSimEngine\.venv\Scripts\python.exe'
    $script = "$PSScriptRoot\big_stack.py"
    $scriptArgs = @("$PSScriptRoot\genesis_profile.py") + $scriptArgs
} else {
    $py = "$Cgq\.venv\Scripts\python.exe"
    $script = "$PSScriptRoot\cgq_profile.py"
    $scriptArgs += @('--cgq', $Cgq)
}
& $ncu --set full --import-source no --nvtx --nvtx-include "step $Frame/" -k "regex:$Kernels" -c $Count `
    --target-processes all @extraArgs -f -o $out $py $script @scriptArgs
exit $LASTEXITCODE
