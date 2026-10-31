#!/bin/bash
# Run all test suites.
# Usage: bash tests/run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=============================================="
echo " soft-sector-data-curation test suite"
echo "=============================================="
echo ""

total_pass=0
total_fail=0

for test_file in "$SCRIPT_DIR"/test_*.py; do
    name="$(basename "$test_file")"
    echo "── $name ──"
    if output=$(python3 "$test_file" 2>&1); then
        echo "$output"
        last_line=$(echo "$output" | tail -1)
        if [[ "$last_line" =~ ^([0-9]+)/([0-9]+)\ passed ]]; then
            total_pass=$((total_pass + BASH_REMATCH[1]))
            total_fail=$((total_fail + BASH_REMATCH[2] - BASH_REMATCH[1]))
        fi
    else
        echo "$output"
        last_line=$(echo "$output" | tail -1)
        if [[ "$last_line" =~ ^([0-9]+)/([0-9]+)\ passed ]]; then
            total_pass=$((total_pass + BASH_REMATCH[1]))
            total_fail=$((total_fail + BASH_REMATCH[2] - BASH_REMATCH[1]))
        else
            total_fail=$((total_fail + 1))
        fi
    fi
    echo ""
done

echo "=============================================="
echo " Total: ${total_pass} passed, ${total_fail} failed"
echo "=============================================="

exit $(( total_fail > 0 ? 1 : 0 ))
