# Genesis side of the cloth parity, on the engine of a Genesis tree with the Quadrants 1.4.0 overlay.
#   powershell -File bench\ipc_cloth\run_genesis.ps1 -Backend gpu|cpu [-Tree <genesis tree>]
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Backend = 'gpu',
    [string]$Tree = 'C:\Users\81946\Projects\GenesisSimEngine\genesis-world-ipc-cloth',
    [string]$Overlay = 'C:\Users\81946\Projects\GenesisSimEngine\scratch\pr19\qd-1.4.0-overlay',
    [int]$N = 10
)
$env:PYTHONPATH = "$Overlay;$Tree"
$cache = 'C:\Users\81946\Projects\GenesisSimEngine\scratch\pr19\cp-run\cache-cloth-parity'
$env:QD_OFFLINE_CACHE_FILE_PATH = "$cache\quadrants"
$env:GS_CACHE_FILE_PATH = "$cache\genesis"
& C:\Users\81946\Projects\GenesisSimEngine\.venv\Scripts\python.exe "$PSScriptRoot\genesis_side.py" --backend $Backend --n $N
exit $LASTEXITCODE
