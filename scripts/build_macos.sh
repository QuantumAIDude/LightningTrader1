#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 build_release.py --prepare-only
python3 build_release.py --clean
