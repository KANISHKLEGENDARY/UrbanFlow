# UrbanFlow — local development startup (Windows PowerShell)
# Installs deps, starts PostgreSQL via Docker when available, initializes DB, runs API.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "`n[UrbanFlow] Installing Python dependencies..." -ForegroundColor Cyan
python -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "[UrbanFlow] Created .env from .env.example — review DATABASE_URL and AUTH_ENABLED" -ForegroundColor Yellow
}

# Try Docker PostgreSQL (known credentials: postgres/postgres)
$dockerOk = $false
try {
    docker info 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[UrbanFlow] Starting PostgreSQL via Docker..." -ForegroundColor Cyan
        docker compose up db -d
        Start-Sleep -Seconds 5
        $dockerOk = $true
    }
} catch {
    Write-Host "[UrbanFlow] Docker not available — using local PostgreSQL (set DATABASE_URL in .env)" -ForegroundColor Yellow
}

Write-Host "[UrbanFlow] Initializing database..." -ForegroundColor Cyan
python scripts/init_database.py
if ($LASTEXITCODE -ne 0) {
    if (-not $dockerOk) {
        Write-Host @"

[UrbanFlow] Database setup failed. Options:
  1. Start Docker Desktop, then re-run: .\scripts\start_dev.ps1
  2. Set your local PostgreSQL password in .env:
       DATABASE_URL=postgresql+asyncpg://postgres:YOUR_PASSWORD@localhost:5432/urbanflow
     Then run: python scripts/init_database.py

"@ -ForegroundColor Yellow
    }
    exit 1
}

Write-Host "[UrbanFlow] Starting API server on http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "[UrbanFlow] Start frontend in another terminal: cd frontend && npm run dev`n" -ForegroundColor Green
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
