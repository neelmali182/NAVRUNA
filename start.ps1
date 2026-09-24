$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
Write-Host "NAVRUNA v4.1.2 - Offline Maritime AI Simulation Lab" -ForegroundColor Cyan

function Test-Python($exe) {
    try {
        $v = & $exe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
        if ($v -notmatch '^3\.(11|12|13)$') { return $false }
        return $true
    } catch { return $false }
}

function Test-NavrunaDeps($exe) {
    try {
        & $exe -c "import fastapi, uvicorn, pydantic, numpy, scipy, shapely, torch" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

function Test-NvidiaGpu {
    try {
        & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

function Test-CudaTorch($exe) {
    try {
        & $exe -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

$python = $null
$currentPython = (Get-Command python -ErrorAction SilentlyContinue).Source

# Prefer the currently active Conda/venv interpreter when it is already usable.
if ($currentPython -and (Test-Python $currentPython) -and (Test-NavrunaDeps $currentPython)) {
    $python = $currentPython
}

# Otherwise look for a supported Python launcher installation.
if (-not $python) {
    foreach ($version in @("3.13", "3.12", "3.11")) {
        try {
            $exe = & py -$version -c "import sys; print(sys.executable)" 2>$null | Select-Object -First 1
            if ($exe -and (Test-Path $exe) -and (Test-Python $exe)) {
                $python = $exe
                break
            }
        } catch {}
    }
}

if (-not $python) {
    throw "Python 3.11, 3.12, or 3.13 was not found. Install one of these versions and rerun."
}

Write-Host "Using Python: $python" -ForegroundColor Green

# If the selected interpreter already has the project dependencies, do not create a venv.
# This avoids Windows ensurepip hangs on machines where Python was installed without a healthy bundled pip.
$vp = $python
$venv = Join-Path $root ".venv"
$forceVenv = $env:NAVRUNA_USE_VENV -eq "1"

if ($forceVenv -or -not (Test-NavrunaDeps $python)) {
    $vp = Join-Path $venv "Scripts\python.exe"
    if (-not (Test-Path $vp)) {
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        & $python -m venv $venv --clear
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create the virtual environment. If your Python ensurepip is hanging, run this launcher without NAVRUNA_USE_VENV=1 and use an interpreter with the required packages."
        }
    }

    Write-Host "Installing/refreshing dependencies..." -ForegroundColor Yellow
    & $vp -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('pip') else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Bootstrapping pip in the virtual environment..." -ForegroundColor Yellow
        & $vp -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) {
            throw "This Python installation cannot bootstrap pip. Recreate the venv with a Python installation that includes ensurepip, then rerun."
        }
    }
    & $vp -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
    & $vp -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
}

if (Test-NvidiaGpu) {
    if (-not (Test-CudaTorch $vp)) {
        Write-Host "CUDA GPU detected; installing CUDA-enabled PyTorch..." -ForegroundColor Yellow
        & $vp -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu126
        if ($LASTEXITCODE -ne 0) { throw "CUDA-enabled PyTorch installation failed." }
    }
    $gpuName = & $vp -c "import torch; print(torch.cuda.get_device_name(0))"
    Write-Host "Using GPU: $gpuName" -ForegroundColor Green
} else {
    Write-Host "Using CPU: no NVIDIA GPU detected" -ForegroundColor Yellow
}

# Stop stale NAVRUNA processes before starting the new backend/frontend.
Get-NetTCPConnection -LocalPort 8000,8080 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object { & taskkill.exe /PID $_ /T /F 2>$null | Out-Null }

Write-Host "Starting FastAPI backend..." -ForegroundColor Yellow
Start-Process -FilePath $vp `
    -ArgumentList "-m uvicorn backend.main:app --host 127.0.0.1 --port 8000" `
    -WorkingDirectory $root

Write-Host "Starting local frontend server..." -ForegroundColor Yellow
Start-Process -FilePath $vp `
    -ArgumentList "-m http.server 8080" `
    -WorkingDirectory (Join-Path $root "frontend")

function Wait-ForHttp($url, $name) {
    for ($attempt = 1; $attempt -le 30; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 1
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { return }
        } catch {}
        Start-Sleep -Milliseconds 250
    }
    throw "$name did not become ready at $url."
}

Wait-ForHttp "http://127.0.0.1:8000/api/health" "Backend"
Wait-ForHttp "http://127.0.0.1:8080/" "Frontend"

Write-Host "NAVRUNA: http://127.0.0.1:8080/" -ForegroundColor Green
Write-Host "API docs: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "Weather API: http://127.0.0.1:8000/api/simulation/weather?hours=0" -ForegroundColor Green
Write-Host "Simulation data: offline / synthetic" -ForegroundColor Green

Start-Process "http://127.0.0.1:8080/"
