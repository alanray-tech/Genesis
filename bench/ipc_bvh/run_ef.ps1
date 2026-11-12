# Acceptance of the EF broad phase of a Genesis tree against CGQ (ef_side.py) on every captured scene: Quadrants 1.4.0
# overlay first on PYTHONPATH, then the tree under test, with compile caches of its own per backend.
#   powershell -File bench\ipc_bvh\run_ef.ps1 -Tree <genesis tree> -Backend gpu
#   powershell -File bench\ipc_bvh\run_ef.ps1 -Tree <genesis tree> -Backend cpu -Scenes kimono
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Python = 'C:\Users\81946\Projects\GenesisSimEngine\.venv\Scripts\python.exe',
    [string]$Overlay = 'C:\Users\81946\Projects\GenesisSimEngine\scratch\pr19\qd-1.4.0-overlay',
    [Parameter(Mandatory = $true)][string]$Tree,
    [ValidateSet('gpu', 'cpu')][string]$Backend = 'gpu',
    [string[]]$Scenes = @('two_bunnies', 'cloth_stack', 'wrecking_ball', 'kimono', 'cloth_stack_x3'),
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
)
$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = "$Overlay;$Tree"
$env:QD_OFFLINE_CACHE_FILE_PATH = "$PSScriptRoot\data\cache_ef_$Backend\quadrants"
$env:GS_CACHE_FILE_PATH = "$PSScriptRoot\data\cache_ef_$Backend\genesis"
$failed = @()
# powershell -File hands a list over as one comma-joined string
foreach ($scene in ($Scenes -split ',')) {
    & $Python "$PSScriptRoot\ef_side.py" --scene $scene --backend $Backend @Arguments
    if ($LASTEXITCODE) { $failed += $scene }
}
if ($failed) {
    Write-Output "FAILED scenes: $($failed -join ', ')"
    exit 1
}
Write-Output "All scenes pass on $Backend."
exit 0
