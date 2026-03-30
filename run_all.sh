#!/bin/bash
# run_all.sh — Reproduce all analyses from scratch
# Usage: bash run_all.sh
#
# Prerequisites: conda env create -f environment.yml && conda activate surprise-eeg-benchmark
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Step 0: Download ERP CORE data ==="
python3 src/download_erp_core.py

echo "=== Step 1: Run MMN pipeline ==="
python3 src/run_pipeline.py --task MMN

echo "=== Step 2: Run P3 pipeline ==="
python3 src/run_pipeline.py --task P3

echo "=== Step 3: Apply exclusion criteria ==="
python3 src/preprocessing/exclusion_criteria.py

echo "=== Step 4: Statistical corrections ==="
python3 src/encoding/statistical_corrections.py

echo "=== Step 5: Missing analyses ==="
python3 src/analyses/missing_analyses.py

echo "=== Step 6: Generate figures ==="
python3 src/figures/make_figures.py MMN

echo "=== Step 7: Build manuscript ==="
python3 manuscript/build_manuscript_docx.py

echo "=== DONE ==="
echo "Manuscript: manuscript/Surprise_EEG_Manuscript_with_Figures.docx"
