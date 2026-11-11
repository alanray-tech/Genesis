# The measured pass of the benchmark, to run alone on the GPU (an exclusive lease), after run_cgq.ps1 -Stage capture:
# the bvh.py side on every scene with both schedules, the CGQ side, then the nsys breakdown of the largest frame of
# every scene on the bvh.py side, and the merged tables.
#   powershell -File bench\ipc_bvh\run_measure.ps1
[CmdletBinding(PositionalBinding = $false)]
param([int]$Repeats = 20)
$ErrorActionPreference = 'Stop'
$genesisPython = 'C:\Users\81946\Projects\GenesisSimEngine\.venv\Scripts\python.exe'
& powershell -NoProfile -File "$PSScriptRoot\run_upstream.ps1" -All --repeats $Repeats
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& powershell -NoProfile -File "$PSScriptRoot\run_cgq.ps1" -Stage time -Repeats $Repeats
if ($LASTEXITCODE) { exit $LASTEXITCODE }
foreach ($scene in 'two_bunnies', 'cloth_stack', 'wrecking_ball', 'kimono', 'cloth_stack_x3') {
    foreach ($schedule in 'deferred', 'inline') {
        & powershell -NoProfile -File "$PSScriptRoot\run_upstream.ps1" -Nsys "${scene}_$schedule" --scene $scene `
            --schedule $schedule --nsys-frame -1 --repeats $Repeats
        if ($LASTEXITCODE) { exit $LASTEXITCODE }
    }
}
& $genesisPython "$PSScriptRoot\make_report.py"
exit $LASTEXITCODE
