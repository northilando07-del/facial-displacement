$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:OPENBLAS_NUM_THREADS = '1'
& python "$PSScriptRoot/run_displacement.py" --config "$PSScriptRoot/displacement.json" @args
if ($LASTEXITCODE -ne 0) { throw 'Displacement workflow failed. See the traceback above.' }
