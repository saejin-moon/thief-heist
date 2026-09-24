#!/usr/bin/env bash
# Full paper build pipeline: tables -> architecture figure -> plots -> tectonic compile.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
TECTONIC="${TECTONIC:-$HOME/.cargo/bin/tectonic}"

echo "=== 1. Generating LaTeX tables from benchmark results ==="
(cd "$ROOT" && uv run python "$DIR/scripts/generate_tables.py")
(cd "$ROOT" && uv run python "$DIR/scripts/generate_ablation_table.py")

echo "=== 2. Compiling TikZ architecture diagram ==="
"$TECTONIC" "$DIR/figures/architecture.tex" -o "$DIR/figures"

echo "=== 3. Generating result figures ==="
(cd "$ROOT" && uv run python "$DIR/scripts/generate_plots.py")

echo "=== 4. Compiling NeurIPS paper with Tectonic ==="
cd "$DIR"
"$TECTONIC" main.tex -o "$DIR"

echo "=== 5. Verifying build ==="
if [ -f "$DIR/main.pdf" ]; then
    TOTAL_PAGES=$(pdfinfo "$DIR/main.pdf" | grep "Pages:" | awk '{print $2}')
    SIZE_KB=$(du -k "$DIR/main.pdf" | cut -f1)
    echo "SUCCESS: main.pdf built successfully ($TOTAL_PAGES pages, ${SIZE_KB} KB)."
else
    echo "ERROR: main.pdf was not generated."
    exit 1
fi