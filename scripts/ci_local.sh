#!/usr/bin/env bash
# Runs the same checks as .github/workflows/ci.yml, in the same order, so CI does not catch things first.
# Run it from anywhere, before every commit:   bash scripts/ci_local.sh
#
# Backend job : README results block is current, ruff, pytest (from backend/, with the real model)
# Frontend job: npm run lint (eslint + tsc), npm test, npm run build
# Not run here: the Docker job (needs Docker) and `npm audit` (informational in CI).
# RUN_HUB_TESTS=1 (CI sets it) downloads the model production uses from the Hugging Face Hub; it is skipped here unless you set it:
#   RUN_HUB_TESTS=1 bash scripts/ci_local.sh
set -u
cd "$(dirname "$0")/.."
PY=python
[ -x .venv/Scripts/python.exe ] && PY="$(pwd)/.venv/Scripts/python.exe"
[ -x .venv/bin/python ] && PY="$(pwd)/.venv/bin/python"

fail=0
step() { printf '\n=== %s ===\n' "$1"; shift; if "$@"; then echo "PASS"; else echo "FAIL: $*"; fail=1; fi; }

# the real-model and 100,000-row database tests write to the temp folder; a nearly full drive fails them with "disk is full"
$PY - <<'EOF'
import shutil, tempfile
free = shutil.disk_usage(tempfile.gettempdir()).free / 2**20
if free < 500:
    print(f"WARNING: only {free:.0f} MB free in {tempfile.gettempdir()}: tests that build large temporary databases may skip or fail. "
          "Free space, or point TEMP/TMP at a roomier drive for this run.")
EOF

step "Backend: README results tables match ml/results" $PY scripts/make_readme_results.py --check
step "Backend: ruff check backend" $PY -m ruff check backend
step "Backend: pytest -rs (in backend/, real model)" bash -c "cd backend && \"$PY\" -m pytest -rs -q -p no:cacheprovider"
step "Frontend: lint + type check" bash -c "cd frontend && npm run lint"
step "Frontend: unit tests" bash -c "cd frontend && npm test"
step "Frontend: production build" bash -c "cd frontend && npm run build"

echo
if [ "$fail" -eq 0 ]; then echo "ALL CI-EQUIVALENT CHECKS PASSED"; else echo "SOME CHECKS FAILED: do not commit yet"; fi
exit "$fail"
