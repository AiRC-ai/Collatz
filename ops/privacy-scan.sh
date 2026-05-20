#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${COLLATZ_REPO_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}"
cd "$ROOT_DIR"

PATTERN='(10\.[0-9]{1,3}\.[0-9]{1,3}\.|192\.168\.|172\.16\.|172\.17\.|172\.18\.|/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+|airc1|ryancox|http://10\.|https://10\.|ssh://|ssh [A-Za-z0-9._-]+@)'

if rg -n "$PATTERN" README.md docs src include tests tools ops compose.yaml CMakeLists.txt .gitignore schemas fixtures data/generated/evidence \
  | rg -v '^ops/privacy-scan\.sh:' \
  | rg -v '^src/collatz_web\.cpp:[0-9]+:.*value\.find' \
  | rg -v '^src/collatz_web\.cpp:[0-9]+:.*unsafe_public_value' \
  | rg -v '^src/collatz_evidence_publish\.cpp:[0-9]+:.*text\.find' \
  | rg -v '^src/collatz_evidence_publish\.cpp:[0-9]+:.*unsafe_public_text' \
  | rg -v '^tools/test_evidence_contract\.py:[0-9]+:.*review /Users/example/private/path'; then
  echo "privacy scan failed" >&2
  exit 1
fi

echo "privacy scan passed"
