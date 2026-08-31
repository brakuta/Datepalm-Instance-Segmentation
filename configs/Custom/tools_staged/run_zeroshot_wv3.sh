#!/usr/bin/env bash
# ==========================================================================
# tools_staged/run_zeroshot_wv3.sh   (Stage D, v1)
# --------------------------------------------------------------------------
# Zero-shot evaluation of Stage C checkpoints on the WV-3 TEST set
# (anchors the annotation-budget curves at zero WV-3 annotations).
#
# Reuses each backbone's STAGE C config and redirects only the test loader
# and evaluator to Sat_30cm via --cfg-options -- no eval configs to write.
#
# Input: a manifest file with one "config<TAB-or-space>checkpoint" pair per
# line (use select_stagec_checkpoint.py output to fill the checkpoints):
#   configs/Custom/3_unified_multisource/maskrcnn_r50_stagec.py /workspace/mmdetection/work_dirs/Stage_C/maskrcnn_r50_stagec/best_....pth
#
# Usage:
#   bash tools_staged/run_zeroshot_wv3.sh zeroshot_pairs.txt
# ==========================================================================
set -euo pipefail
# Resolve the mmdetection root from this script's own location
# (configs/Custom/tools_staged/ -> three levels up), so the project
# folder remains relocatable and the script can be called from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${MMDET_ROOT}"
PAIRS=${1:?usage: run_zeroshot_wv3.sh <pairs.txt>}
SAT=/workspace/datasets/COCO/Sat_30cm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while read -r CFG CKPT; do
    [[ -z "${CFG}" || "${CFG}" == \#* ]] && continue
    NAME=$(basename "${CFG}" .py)
    OUT=/workspace/mmdetection/work_dirs/Stage_D/zeroshot/${NAME}
    echo "=== zero-shot WV-3: ${NAME} ==="
    python tools/test.py "${CFG}" "${CKPT}" \
        --work-dir "${OUT}" \
        --cfg-options \
            test_dataloader.dataset.data_root="${SAT}/" \
            test_dataloader.dataset.ann_file=Annotations/test_sat.json \
            test_dataloader.dataset.data_prefix.img=test_sat/ \
            test_evaluator.ann_file="${SAT}/Annotations/test_sat.json" \
        2>&1 | tee "${OUT}.log" | grep -E "mAP|Average" || true
done < "${PAIRS}"
