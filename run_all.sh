#!/bin/bash
# Run test evaluation for all 5 datasets (independent logs & checkpoints)

PYTHON="D:/Projects/conda_envs/mmfrnet/python.exe"
PROJECT="D:/Documents/Python Programs/lab/2026.7.17 论文重投/project"

declare -A EPOCHS
EPOCHS=(
    ["pneumoniamnist"]=30
    ["breastmnist"]=25
    ["retinamnist"]=15
    ["dermamnist"]=15
    ["organcmnist"]=20
)

for DS in pneumoniamnist breastmnist retinamnist dermamnist organcmnist; do
    echo "=============================================="
    echo "Dataset: $DS (epochs=${EPOCHS[$DS]})"
    echo "=============================================="
    cd "$PROJECT"
    $PYTHON -u train.py \
        --dataset "$DS" \
        --batch_size 32 \
        --epochs "${EPOCHS[$DS]}" \
        --save_dir "checkpoints/$DS" \
        > "log/${DS}_test.log" 2>&1
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
    grep -E "(Best val|Test )" "log/${DS}_test.log"
done
