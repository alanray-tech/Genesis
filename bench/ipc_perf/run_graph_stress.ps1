# Run several graph_stress.py processes at once on the instrumented dev-fast overlay, each logging to results\.
#   powershell -File bench\ipc_perf\run_graph_stress.ps1 [-Processes 3] [-Rounds 1000]
[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Processes = 3,
    [int]$Rounds = 1000,
    [string]$Overlay = 'C:\Users\81946\Projects\GenesisSimEngine\scratch\pr19\qd-devfast-dbg-overlay'
)
$env:PYTHONPATH = $Overlay
$py = 'C:\Users\81946\Projects\GenesisSimEngine\.venv\Scripts\python.exe'
$results = "$PSScriptRoot\results"
$running = foreach ($i in 1..$Processes) {
    Start-Process -FilePath $py -ArgumentList "$PSScriptRoot\graph_stress.py", '--rounds', $Rounds -NoNewWindow `
        -PassThru -RedirectStandardOutput "$results\graph_stress_$i.log" -RedirectStandardError "$results\graph_stress_$i.err"
}
$running | Wait-Process
foreach ($i in 1..$Processes) {
    Get-Content "$results\graph_stress_$i.log" -Tail 1
    Get-Content "$results\graph_stress_$i.err" | Select-String -Pattern 'Error|error' | Select-Object -First 3
}
