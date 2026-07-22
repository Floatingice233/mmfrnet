#!/bin/bash
# Run full 150-epoch training for all 5 datasets (slowest first)

PYTHON="D:/Projects/conda_envs/mmfrnet/python.exe"
PROJECT="D:/Documents/Python Programs/lab/2026.7.17 论文重投/project"

declare -A EPOCHS
EPOCHS=(
    ["organcmnist"]=150
    ["dermamnist"]=150
    ["pneumoniamnist"]=150
    ["retinamnist"]=150
    ["breastmnist"]=150
)

for DS in organcmnist dermamnist pneumoniamnist retinamnist breastmnist; do
    echo "=============================================="
    echo "Dataset: $DS (epochs=${EPOCHS[$DS]})"
    echo "=============================================="
    cd "$PROJECT"
    mkdir -p "logs/$DS"
    TS=$(date +%Y%m%d_%H%M%S)
    $PYTHON -u scripts/train.py \
        --dataset "$DS" \
        --epochs "${EPOCHS[$DS]}" \
        --save_dir "checkpoints/$DS" \
        --num_workers 0 \
        > "logs/${DS}/train_${TS}.log" 2>&1
    echo "Done: $DS"
    echo ""
done

echo "All done!"
# Print summary
cd "$PROJECT"
echo ""
echo "===== SUMMARY ====="
for DS in organcmnist dermamnist pneumoniamnist retinamnist breastmnist; do
    echo "--- $DS ---"
    grep -E "(Best val|Test )" "logs/${DS}/"*.log 2>/dev/null || echo "  (no log found)"
done
