# =====================================================================
# git_commit_push.ps1
# Commits & pushes the current state of the GUI_Abaqus repo.
#
# Usage (in PowerShell, from C:\GUI_Abaqus):
#     .\git_commit_push.ps1
# or with a custom message:
#     .\git_commit_push.ps1 -Message "your commit message"
# =====================================================================
param(
    [string]$Message = "Step tab + mass scaling + working Run button"
)

$ErrorActionPreference = "Stop"

# Move to the script's directory (the repo root)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
Write-Host "Repository: $scriptDir" -ForegroundColor DarkGray

# Sanity check
if (-not (Test-Path ".\.git")) {
    Write-Host "Not a git repository. Run git_init.ps1 first." -ForegroundColor Red
    exit 1
}

# Show what's about to be committed
Write-Host "==> Current changes:" -ForegroundColor Cyan
git status --short
Write-Host ""

# Stage everything
Write-Host "==> Staging changes..." -ForegroundColor Cyan
git add .

# Commit if there's anything to commit
$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "Nothing to commit. Working tree clean." -ForegroundColor Yellow
    exit 0
}

Write-Host "==> Committing with message:" -ForegroundColor Cyan
Write-Host "    `"$Message`"" -ForegroundColor DarkGray
git commit -m $Message | Out-Null

# Push if a remote 'origin' exists
$remote = git remote
if ($remote -contains "origin") {
    Write-Host "==> Pushing to origin..." -ForegroundColor Cyan
    git push origin (git rev-parse --abbrev-ref HEAD)
} else {
    Write-Host "==> No 'origin' remote configured — skipping push." -ForegroundColor Yellow
    Write-Host "    To add one:  git remote add origin <url>" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Done. Latest commits:" -ForegroundColor Green
git log --oneline -n 5
