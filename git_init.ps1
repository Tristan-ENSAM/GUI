# =====================================================================
# git_init.ps1
# Initialise the GUI_Abaqus repository on Windows.
#
# Usage (in PowerShell, from C:\GUI_Abaqus):
#     .\git_init.ps1
# If execution policy blocks the script:
#     powershell -ExecutionPolicy Bypass -File .\git_init.ps1
# =====================================================================

$ErrorActionPreference = "Stop"

Write-Host "==> Checking git is available..." -ForegroundColor Cyan
try {
    git --version | Out-Null
} catch {
    Write-Host "git is not on PATH. Install Git for Windows first." -ForegroundColor Red
    exit 1
}

# Make sure the script runs from its own directory (the project root)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
Write-Host "Working directory: $scriptDir" -ForegroundColor DarkGray

# Sanity check: must contain the gui/ folder
if (-not (Test-Path ".\gui")) {
    Write-Host "No gui/ folder here. Run this script from the project root." -ForegroundColor Red
    exit 1
}

# Init only if not already a repo
if (-not (Test-Path ".\.git")) {
    Write-Host "==> git init..." -ForegroundColor Cyan
    git init -b main | Out-Null
} else {
    Write-Host "==> Repository already initialised, skipping git init" -ForegroundColor Yellow
}

# .gitignore should already be there from the delivery, but write it if missing
if (-not (Test-Path ".\.gitignore")) {
    Write-Host ".gitignore missing — copy it from the delivery before running this script." -ForegroundColor Red
    exit 1
}

# Configure user.name / user.email locally if not set globally
$userName = git config --get user.name
$userEmail = git config --get user.email
if ([string]::IsNullOrEmpty($userName)) {
    Write-Host "==> No global user.name set, configuring local one." -ForegroundColor Cyan
    git config user.name  "Tristan Chenevez"
}
if ([string]::IsNullOrEmpty($userEmail)) {
    Write-Host "==> No global user.email set, configuring local one." -ForegroundColor Cyan
    git config user.email "tristan.chenevez@example.invalid"
    Write-Host "    (Update with: git config user.email <your-email>)" -ForegroundColor DarkGray
}

Write-Host "==> Staging files..." -ForegroundColor Cyan
git add .

Write-Host "==> Committing..." -ForegroundColor Cyan
git commit -m "Initial commit: Qt-based Abaqus pre-processor GUI" | Out-Null

Write-Host ""
Write-Host "Done. Repository state:" -ForegroundColor Green
git log --oneline -n 5
Write-Host ""
Write-Host "Next steps (optional, to push to a remote):" -ForegroundColor Cyan
Write-Host "  git remote add origin <https-or-ssh-url>"
Write-Host "  git push -u origin main"
