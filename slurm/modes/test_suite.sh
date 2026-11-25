#!/usr/bin/env bash
# The test suite, on a node.
#   PYTHON=... [PYTHONPATH=...] sbatch slurm/modes/test_suite.sh [pytest arguments]
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1:30:00
#SBATCH --job-name=mlgw-tests
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
PYTHONPATH=".:${PYTHONPATH:-}" "$PYTHON" -m pytest -q "$@"
