# Run a script of bench/ipc_rigid on the engine of a Genesis tree with the Quadrants 1.4.0 overlay.
#   powershell -File bench\ipc_rigid\run.ps1 -Script se3_conformance.py [-Tree <genesis tree>] -- <script arguments>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Script,
    [string]$Tree = 'C:\Users\81946\Projects\GenesisSimEngine\genesis-world-ipc-rigid',
    [string]$Overlay = 'C:\Users\81946\Projects\GenesisSimEngine\scratch\pr19\qd-1.4.0-overlay',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$ScriptArgs
)
$env:PYTHONPATH = "$Overlay;$Tree"
$cache = 'C:\Users\81946\Projects\GenesisSimEngine\scratch\pr19\cp-run\cache-rigid-parity'
$env:QD_OFFLINE_CACHE_FILE_PATH = "$cache\quadrants"
$env:GS_CACHE_FILE_PATH = "$cache\genesis"
& C:\Users\81946\Projects\GenesisSimEngine\.venv\Scripts\python.exe "$PSScriptRoot\$Script" @ScriptArgs
exit $LASTEXITCODE
