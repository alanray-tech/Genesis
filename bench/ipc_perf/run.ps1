# Run a script of bench/ipc_perf on the engine of a Genesis tree with the instrumented dev-fast overlay, optionally under nsys.
#   powershell -File bench\ipc_perf\run.ps1 -Script genesis_profile.py [-Tree <genesis tree>] [-Nsys <report path>] -- <args>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Script,
    [string]$Tree = 'C:\Users\81946\Projects\GenesisSimEngine\genesis-world-ipc-perf',
    [string]$Overlay = 'C:\Users\81946\Projects\GenesisSimEngine\scratch\pr19\qd-devfast-dbg-overlay',
    [string]$Nsys = '',
    [string]$Cache = 'C:\Users\81946\Projects\GenesisSimEngine\scratch\pr19\cp-run\cache-ipc-perf',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ScriptArgs
)
$env:PYTHONPATH = "$Overlay;$Tree"
$cache = $Cache
$env:QD_OFFLINE_CACHE_FILE_PATH = "$cache\quadrants"
$env:GS_CACHE_FILE_PATH = "$cache\genesis"
$py = 'C:\Users\81946\Projects\GenesisSimEngine\.venv\Scripts\python.exe'
if ($Nsys) {
    & nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end=stop `
        --force-overwrite=true -o $Nsys $py "$PSScriptRoot\$Script" @ScriptArgs
} else {
    & $py "$PSScriptRoot\$Script" @ScriptArgs
}
exit $LASTEXITCODE
