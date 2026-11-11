# Upstream side of the benchmark (upstream_side.py) on the Genesis environment: Quadrants 1.4.0 overlay first on
# PYTHONPATH, then this worktree, with compile caches of its own.
#   powershell -File bench\ipc_bvh\run_upstream.ps1 --scene kimono --schedule deferred
#   powershell -File bench\ipc_bvh\run_upstream.ps1 -All
#   powershell -File bench\ipc_bvh\run_upstream.ps1 -Nsys kimono_deferred --scene kimono --schedule deferred --nsys-frame -1
# -All runs every scene with both schedules (the measured pass). -Nsys traces the run with nsys into
# data\nsys\upstream_<name> and parses it with upstream_nsys.py. Other arguments go to upstream_side.py, or to the
# script of this folder named first (hit_distribution.py --scene kimono).
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Python = 'C:\Users\81946\Projects\GenesisSimEngine\.venv\Scripts\python.exe',
    [string]$Overlay = 'C:\Users\81946\Projects\GenesisSimEngine\scratch\pr19\qd-1.4.0-overlay',
    [switch]$All,
    [string]$Nsys = '',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
)
$ErrorActionPreference = 'Stop'
$tree = (Resolve-Path "$PSScriptRoot\..\..").Path
$env:PYTHONPATH = "$Overlay;$tree"
$env:QD_OFFLINE_CACHE_FILE_PATH = "$PSScriptRoot\data\cache\quadrants"
$env:GS_CACHE_FILE_PATH = "$PSScriptRoot\data\cache\genesis"
if ($All) {
    foreach ($scene in 'two_bunnies', 'cloth_stack', 'wrecking_ball', 'kimono', 'cloth_stack_x3') {
        foreach ($schedule in 'deferred', 'inline') {
            & $Python "$PSScriptRoot\upstream_side.py" --scene $scene --schedule $schedule @Arguments
            if ($LASTEXITCODE) { exit $LASTEXITCODE }
        }
    }
    exit 0
}
if ($Nsys) {
    $trace = "$PSScriptRoot\data\nsys\upstream_$Nsys"
    New-Item -ItemType Directory -Force "$PSScriptRoot\data\nsys" | Out-Null
    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop -t cuda,nvtx --force-overwrite true `
        -o $trace $Python "$PSScriptRoot\upstream_side.py" @Arguments
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    nsys export --type sqlite --force-overwrite true -o "$trace.sqlite" "$trace.nsys-rep"
    $scene = $Arguments[[array]::IndexOf($Arguments, '--scene') + 1]
    $schedule = $Arguments[[array]::IndexOf($Arguments, '--schedule') + 1]
    & $Python "$PSScriptRoot\upstream_nsys.py" --scene $scene --schedule $schedule "$trace.sqlite"
    exit $LASTEXITCODE
}
$script = 'upstream_side.py'
if ($Arguments -and $Arguments[0] -like '*.py') {
    $script, $Arguments = $Arguments[0], @($Arguments | Select-Object -Skip 1)
}
& $Python "$PSScriptRoot\$script" @Arguments
exit $LASTEXITCODE
