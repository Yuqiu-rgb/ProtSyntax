#!/usr/bin/env bash
set -euo pipefail

OWNER="${1:-Yuqiu-rgb}"
REPO="${2:-ProtSyntax}"
REMOTE_URL="https://github.com/${OWNER}/${REPO}.git"

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI is required. Install it, then run: gh auth login"
  exit 1
fi

gh auth status >/dev/null

if gh repo view "${OWNER}/${REPO}" >/dev/null 2>&1; then
  echo "Repository ${OWNER}/${REPO} already exists."
else
  gh repo create "${OWNER}/${REPO}" \
    --public \
    --description "ProtSyntax: a PTM-aware protein language model for decoding post-translational modification syntax and function" \
    --source . \
    --remote origin \
    --push
  exit 0
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "${REMOTE_URL}"
else
  git remote add origin "${REMOTE_URL}"
fi

# GitHub-created repositories often contain a seed commit from a selected
# README/license/gitignore. The local release history is authoritative here,
# but keep the lease explicit so we only overwrite the ref we inspected.
REMOTE_MAIN_SHA="$(git ls-remote --heads origin main | awk '{print $1}')"
if [[ -n "${REMOTE_MAIN_SHA}" ]]; then
  git push -u origin main --force-with-lease="refs/heads/main:${REMOTE_MAIN_SHA}"
else
  git push -u origin main
fi
