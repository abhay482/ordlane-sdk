#!/usr/bin/env bash
# Run routing demo from repo root (used by VHS tape and local recording).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src"
exec python3 examples/demo_routing.py
