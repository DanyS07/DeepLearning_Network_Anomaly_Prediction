#!/bin/bash
# Runs Stages 1-7 in order and stops at the first error.
set -e
cd "$(dirname "$0")"
for s in 01_data_cleaning 02_feature_engineering 03_future_labels \
         04_sequence_generation 05_cv_folds 06_feature_scaling \
         07_model_ready_validation; do
    echo "================ $s ================"
    python3 preprocessing/$s.py
done
