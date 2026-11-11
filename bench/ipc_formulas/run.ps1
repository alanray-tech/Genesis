# IPC formula parity, end to end: inputs, CGQ reference (nvcc with and without contraction), Genesis side (CUDA and
# CPU; fast_math off, fast_math on, and Eigen's summation order), the 50-digit referee, then the tables.
#   powershell -File bench\ipc_formulas\run.ps1
#   powershell -Command "& bench\ipc_formulas\run.ps1 -Stages genesis,compare"
# $Cgq is a checkout of the ground truth commit (cases.GROUND_TRUTH), built with the interpreter of an existing CGQ
# environment ($CgqPython). Inputs, outputs and caches go to data\<commit>, tables to results\<commit>. The acceptance
# table is table_acceptance.md: Genesis with fast_math off against CGQ as production builds it (nvcc default
# --fmad=true), each row passing by pass1, order or referee (see compare.py). The other tables are information.
[CmdletBinding(PositionalBinding = $false)]
param(
    [string[]]$Stages = @('inputs', 'cgq', 'genesis', 'compare'),
    [string]$Cgq = 'C:\Users\81946\Projects\CudaGraphQIPC\cuda-graph-qipc-ref-aeb58956',
    [string]$CgqPython = 'C:\Users\81946\Projects\CudaGraphQIPC\cuda-graph-qipc-bvh-bench\.venv\Scripts\python.exe',
    [string]$Library = 'C:\Users\81946\Projects\GenesisSimEngine\genesis-world-ipc-formulas',
    [string]$Overlay = 'C:\Users\81946\Projects\GenesisSimEngine\scratch\pr19\qd-1.4.0-overlay',
    [string]$GenesisPython = 'C:\Users\81946\Projects\GenesisSimEngine\.venv\Scripts\python.exe',
    [string]$GroundTruth = 'aeb58956'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$data = "$PSScriptRoot\data\$GroundTruth"

function Invoke-Checked([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE) { throw "$exe $arguments failed with $LASTEXITCODE" }
}

if ($Stages -contains 'inputs') {
    Invoke-Checked $GenesisPython @('inputs.py')
}
if ($Stages -contains 'cgq') {
    foreach ($fmad in 'false', 'true') {
        Invoke-Checked $CgqPython @('cgq_side.py', '--cgq', $Cgq, '--fmad', $fmad)
    }
}
if ($Stages -contains 'genesis') {
    $env:PYTHONPATH = "$Overlay;$Library"
    # (fast_math, summation order): the parity runs, the runs of Genesis' default fast_math, and the Eigen-order runs
    # of the order clause. The offline caches key on the source alone, so each setting keeps its own.
    foreach ($setting in @(@('false', 'quadrants'), @('true', 'quadrants'), @('false', 'eigen'))) {
        $fastMath, $sumOrder = $setting
        $env:QD_OFFLINE_CACHE_FILE_PATH = "$data\cache-fast-math-$fastMath-$sumOrder\quadrants"
        $env:GS_CACHE_FILE_PATH = "$data\cache-fast-math-$fastMath-$sumOrder\genesis"
        foreach ($backend in 'cuda', 'cpu') {
            Invoke-Checked $GenesisPython @('genesis_side.py', '--backend', $backend, '--fast-math', $fastMath,
                '--sum-order', $sumOrder)
        }
    }
}
if ($Stages -contains 'compare') {
    $runs = @('cuda_fast-math-false', 'cpu_fast-math-false')
    $rule = @('--reference', 'fmad-true', '--ieee', 'fmad-false', '--order', '_sum-order-eigen')
    # The pass1 and order clauses alone, whose failing rows go to the 50-digit referee
    & $GenesisPython compare.py @rule --label acceptance-noreferee --runs @runs
    Invoke-Checked $GenesisPython @('referee.py', '--summary', 'acceptance-noreferee')
    # Information, whatever their outcome: both IEEE evaluations at 1e-12 alone, Genesis with fast_math on (pass1
    # clause), and CGQ's production build against its IEEE build at 1e-12
    & $GenesisPython compare.py --reference fmad-false --label ieee-strict --runs @runs
    & $GenesisPython compare.py --reference fmad-true --ieee fmad-false --label genesis-fast-math --runs `
        cuda_fast-math-true cpu_fast-math-true
    & $GenesisPython compare.py --reference fmad-false --label cgq-contraction --reference-run --runs fmad-true
    # Acceptance (compare.py): pass1, order or referee on every row, CUDA and CPU; then the per-function clause counts
    # and the rows where Genesis is closer to the exact value than CGQ
    & $GenesisPython compare.py @rule --referee "$data\referee.npz" --label acceptance --runs @runs
    $acceptance = $LASTEXITCODE
    Invoke-Checked $GenesisPython @('make_report.py', '--label', 'acceptance')
    Invoke-Checked $GenesisPython @('cgq_accuracy.py')
    if ($acceptance) { throw "acceptance failed" }
}
