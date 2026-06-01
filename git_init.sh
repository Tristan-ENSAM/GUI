#!/usr/bin/env bash
# =====================================================================
# git_init.sh — Initialise the GUI_Abaqus repository (Linux / WSL).
#
# Usage:
#     chmod +x git_init.sh
#     ./git_init.sh
# =====================================================================
set -euo pipefail

# Move to the script's own directory
cd "$(dirname "$0")"

echo "==> Checking git is available..."
command -v git >/dev/null 2>&1 || { echo "git not found on PATH"; exit 1; }

echo "Working directory: $(pwd)"

# Sanity check
if [ ! -d "./gui" ]; then
    echo "No gui/ folder here. Run this script from the project root."
    exit 1
fi

# Init only if not already a repo
if [ ! -d "./.git" ]; then
    echo "==> git init..."
    git init -b main >/dev/null
else
    echo "==> Repository already initialised, skipping git init"
fi

if [ ! -f "./.gitignore" ]; then
    echo ".gitignore missing — copy it from the delivery before running."
    exit 1
fi

# Configure user.name / user.email locally if not set globally
if [ -z "$(git config --get user.name || true)" ]; then
    echo "==> No global user.name set, configuring local one."
    git config user.name  "Tristan Chenevez"
fi
if [ -z "$(git config --get user.email || true)" ]; then
    echo "==> No global user.email set, configuring local one."
    git config user.email "tristan.chenevez@example.invalid"
    echo "    (Update with: git config user.email <your-email>)"
fi

echo "==> Staging files..."
git add .

echo "==> Committing..."
git commit -m "Initial commit: Qt-based Abaqus pre-processor GUI" >/dev/null

echo
echo "Done. Repository state:"
git log --oneline -n 5
echo
echo "Next steps (optional):"
echo "  git remote add origin <https-or-ssh-url>"
echo "  git push -u origin main"
