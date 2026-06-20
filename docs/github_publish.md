# GitHub Publish Notes

Target repository:

```text
https://github.com/Yuqiu-rgb/ProtSyntax
```

This local repository already has a clean `main` branch and an `origin` remote configured for the target URL.

## Preferred One-Step Publish

After installing and authenticating the GitHub CLI:

```bash
gh auth login
bash scripts/publish_to_github.sh
```

The script will:

1. create `Yuqiu-rgb/ProtSyntax` if it does not already exist;
2. keep the repository public;
3. set `origin` to the target URL;
4. push the local `main` branch.

## Manual Publish

If the repository is created through the GitHub website first, run:

```bash
git remote set-url origin https://github.com/Yuqiu-rgb/ProtSyntax.git
git push -u origin main
```
