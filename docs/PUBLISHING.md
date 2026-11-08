# Proposed Git procedure (not executed)

Use only the reviewed publication directory. Do not run these steps from the original dataset workspace. Obtain author approval before any initialization, commit or push.

```sh
# First change into the reviewed publication directory.
git init -b main
git remote add origin https://github.com/KeweKiwi/multimodal-gambling-robustness.git
python scripts/verify_publication.py
git add -- .gitignore README.md MANIFEST.md MANIFEST.csv SHA256SUMS requirements.txt requirements-experiment.txt docs notebooks scripts src splits perturbations results models provenance
git diff --cached --check
git diff --cached --stat
git status --short
# Confirm the GitHub repository still has no branch heads before first publication.
git ls-remote --heads origin
# Stop and inspect if the preceding command prints any branch heads.
git commit -m "Publish audited FINAL V2 research artifacts for JOSCEX 338"
git push -u origin main
```

The separate local handoff document provides the exact local `cd` command. No inherited dataset commit is required. No force push is proposed. The staged paths omit `work/` and any runtime data. If the author chooses a license, update the docs and regenerate the file inventory before staging. Retain the complete prediction exports; a summaries-only release cannot replay the reported statistical results.
