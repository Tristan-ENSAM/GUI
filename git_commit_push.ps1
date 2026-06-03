# =====================================================================
# git_commit_push.ps1
# Commits and pushes the current state of the GUI_Abaqus repo.
#
# Usage (in PowerShell, from C:\GUI_Abaqus):
#     powershell -ExecutionPolicy Bypass -File .\git_commit_push.ps1
#     powershell -ExecutionPolicy Bypass -File .\git_commit_push.ps1 -Message "your message"
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

# Verify that git is available on PATH. PowerShell otherwise spits a
# generic "term not recognized" error.
$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    Write-Host "git is not on PATH in this console." -ForegroundColor Red
    Write-Host "Open a Git Bash, a regular PowerShell, or run from a console" -ForegroundColor DarkGray
    Write-Host "where Git for Windows was added to PATH." -ForegroundColor DarkGray
    exit 1
}

# Show what is about to be committed
Write-Host "==> Current changes:" -ForegroundColor Cyan
git status --short
Write-Host ""

# Stage everything
Write-Host "==> Staging changes..." -ForegroundColor Cyan
git add .

# Commit only if there is something staged
$staged = git diff --cached --name-only
if (-not $staged) {
    Write-Host "Nothing to commit. Working tree clean." -ForegroundColor Yellow
    exit 0
}

Write-Host "==> Committing with message:" -ForegroundColor Cyan
Write-Host "    $Message" -ForegroundColor DarkGray
git commit -m $Message | Out-Null

# Push if a remote 'origin' exists
$remotes = git remote
if ($remotes -contains "origin") {
    $branch = git rev-parse --abbrev-ref HEAD
    Write-Host "==> Pushing to origin/$branch ..." -ForegroundColor Cyan
    git push origin $branch
} else {
    Write-Host "==> No 'origin' remote configured. Skipping push." -ForegroundColor Yellow
    # Single-quoted string here so PowerShell does not try to parse `<url>`
    # as a redirection operator. The text is just displayed as a hint.
    Write-Host '    To add one:  git remote add origin URL_OF_REMOTE' -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Done. Latest commits:" -ForegroundColor Green
git log --oneline -n 5
