#!/bin/bash
# Run test evaluation for all 5 datasets (independent logs & checkpoints)

PYTHON="D:/Projects/conda_envs/mmfrnet/python.exe"
PROJECT="D:/Documents/Python Programs/lab/2026.7.17 论文重投/project"

declare -A EPOCHS
EPOCHS=(
    ["pneumoniamnist"]=50
    ["breastmnist"]=50
    ["retinamnist"]=50
    ["dermamnist"]=50
    ["organcmnist"]=50
)

for DS in pneumoniamnist breastmnist retinamnist dermamnist organcmnist; do
    echo "=============================================="
    echo "Dataset: $DS (epochs=${EPOCHS[$DS]})"
    echo "=============================================="
    cd "$PROJECT"
    mkdir -p "logs/$DS"
    TS=$(date +%Y%m%d_%H%M%S)
    $PYTHON -u train.py \
        --dataset "$DS" \
        --batch_size 32 \
        --epochs "${EPOCHS[$DS]}" \
        --save_dir "checkpoints/$DS" \
        > "logs/${DS}/train_${TS}.log" 2>&1
    echo "Done: $DS"
    echo ""
done

echo "All done!"
# Print summary
cd "$PROJECT"
echo ""
echo "===== SUMMARY ====="
for DS in pneumoniamnist breastmnist retinamnist dermamnist organcmnist; do
    echo "--- $DS ---"
    grep -E "(Best val|Test )" "logs/${DS}/"*.log 2>/dev/null || echo "  (no log found)"
done
