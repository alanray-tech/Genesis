# The whole cloth parity of Genesis with CGQ: CGQ's reference runs, then Genesis on CPU and GPU.
#   powershell -File bench\ipc_cloth\run_all.ps1 [-Tree <genesis tree>] [-Cgq <cuda-graph-qipc checkout>]
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Tree = 'C:\Users\81946\Projects\GenesisSimEngine\genesis-world-ipc-cloth',
    [string]$Cgq = 'C:\Users\81946\Projects\CudaGraphQIPC\cuda-graph-qipc-p2a-aeb58956',
    [int]$N = 10
)
& "$Cgq\.venv\Scripts\python.exe" "$PSScriptRoot\cgq_side.py" --n $N
if ($LASTEXITCODE) { exit $LASTEXITCODE }
foreach ($backend in 'cpu', 'gpu') {
    powershell -File "$PSScriptRoot\run_genesis.ps1" -Backend $backend -Tree $Tree -N $N
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
}
