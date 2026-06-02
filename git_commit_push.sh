#!/usr/bin/env bash
# =====================================================================
# git_commit_push.sh — Commits & pushes the current state of the repo.
#
# Usage:
#     chmod +x git_commit_push.sh
#     ./git_commit_push.sh                          # default message
#     ./git_commit_push.sh "my commit message"     # custom message
# =====================================================================
set -euo pipefail

MSG="${1:-Step tab + mass scaling + working Run button}"

cd "$(dirname "$0")"
echo "Repository: $(pwd)"

if [ ! -d ".git" ]; then
    echo "Not a git repository. Run ./git_init.sh first."
    exit 1
fi

echo "==> Current changes:"
git status --short
echo

echo "==> Staging..."
git add .

if git diff --cached --quiet; then
    echo "Nothing to commit. Working tree clean."
    exit 0
fi

echo "==> Committing: \"$MSG\""
git commit -m "$MSG" >/dev/null

if git remote | grep -qx "origin"; then
    branch="$(git rev-parse --abbrev-ref HEAD)"
    echo "==> Pushing to origin/$branch..."
    git push origin "$branch"
else
    echo "==> No 'origin' remote configured — skipping push."
    echo "    To add one:  git remote add origin <url>"
fi

echo
echo "Done. Latest commits:"
git log --oneline -n 5
