#!/usr/bin/env bash
# ==========================================================================
# tools_staged/run_staged_matrix.sh   (Stage D, v4)
# --------------------------------------------------------------------------
# Runs the Stage D v4 matrix: 4 backbones x 2 arms on the refined
# WorldView-3 512 px tiling.
#
#   b0 : ImageNet -> FULL TRAINING on WV-3
#        config maskrcnn_<bb>_staged_full.py
#        nothing frozen, lr 1e-4, backbone lr_mult 0.1, 60k iters x batch 2
#        = 120,000 samples (the same exposure as Stage C). NO load_from.
#
#   c  : Stage C best_GE prior -> FINE-TUNE on WV-3
#        config maskrcnn_<bb>_staged_ft.py
#        frozen_stages=2, head lr 2e-5, backbone lr_mult 0.01, 40k iters.
#        load_from is the ONLY thing injected here.
#
#   cf : Stage C best_GE prior -> FULL TRAINING on WV-3
#        config maskrcnn_<bb>_staged_full.py + load_from
#        Identical recipe to b0. This is the arm that isolates the PRIOR,
#        because it holds the recipe fixed; b0-vs-c confounds the two.
#
#   s  : GE-30sim prior -> FULL TRAINING on WV-3
#        config maskrcnn_<bb>_staged_full.py + load_from (Simulated/ stage 1)
#        Same recipe again, so s, cf and b0 differ ONLY in initialisation.
#
# WHY ARM s CAME BACK IN v5
#   The b0-vs-cf result was 3/3 against the Stage C prior: ImageNet won every
#   matched pair on peak (+0.007 to +0.009) and by more on the plateau mean
#   (+0.016 to +0.060), and the prior runs were 1.5-2x noisier in all three.
#   Spatial-Mamba cf peaked at 15,600 and then collapsed outright, ending at
#   0.581 and still falling.
#
#   The diagnosis is SCALE, not domain. Stage C learned crowns at 40-120 px;
#   WorldView-3 crowns are ~20 px, so the prior encodes features at the wrong
#   spatial frequency and the run is spent fighting its own initialisation.
#   GE-30sim does not have that defect -- it is Google Earth imagery resampled
#   to 30 cm, so its crowns are already ~20 px. If a prior transfers when the
#   RESOLUTION matches and fails when only the DOMAIN matches, that is the
#   cross-resolution thesis demonstrated rather than asserted, and it is worth
#   four runs.
#
#   Arm s deliberately reuses the b0 config. Giving it a recipe of its own
#   would rebuild exactly the confound that made arm c uninterpretable.
#
# WHY cf EXISTS, AND WHY IT MATTERS MORE THAN c
#   First result, ConvNeXt-T: b0 reached 0.8020 segm mAP@50 (best at 32,400,
#   early-stopped ~46,800) while c reached only 0.7690 (best at 28,800,
#   stopped ~38,400). The prior LOST by 3.3 points -- and the cost argument
#   went with it, since both runs took 5-7 hours, so the fine-tune is not
#   meaningfully cheaper either.
#
#   That is not evidence the prior fails to transfer. Arm c freezes the stem
#   and first two stages and moves the rest at 2e-5 x 0.01, a recipe built for
#   a near-domain prior. Stage C learned crowns at 40-120 px; WorldView-3
#   crowns span ~20 px. Freezing the early layers is precisely what prevents
#   the model rescaling the features that are wrong. Arm cf removes that
#   explanation: same weights, same recipe as b0, so a remaining gap is the
#   prior and nothing else.
#
# WHY b0 CHANGED CONFIG IN v4
#   v3 ran arm b0 through *_staged_ft.py with load_from omitted, i.e. the
#   locked fine-tune recipe applied to ImageNet weights: stem and first two
#   stages frozen, backbone moving at 2e-5 x 0.01. That recipe assumes a
#   near-domain prior; on ImageNet weights it underfits, and beating an
#   underfit baseline would prove nothing about the value of the Stage C
#   prior. v4 gives arm b0 a genuine training run in its own config. Running
#   the old path would silently reinstate the strawman, so this script no
#   longer offers it.
#
# ARM bu IS STILL CUT
#   The budget curve is an annotation-cost study in its own right and belongs
#   to the companion paper -- see maskrcnn_palm_staged/STAGE_D_README.md
#   section 4. This script refuses it rather than letting a cut arm run out
#   of habit. Arm s still owes the paper a Methods paragraph describing how
#   GE-30sim was produced and why it is a fair proxy; without that paragraph
#   the arm is not reportable, however the numbers come out.
#
# USAGE (invoke by its real path; the script cd's to the repo root itself):
#   bash configs/Custom/tools_staged/run_staged_matrix.sh <backbone|all> <b0|c|cf|s|all>
# Examples:
#   bash configs/Custom/tools_staged/run_staged_matrix.sh all b0
#   bash configs/Custom/tools_staged/run_staged_matrix.sh all s
#   bash configs/Custom/tools_staged/run_staged_matrix.sh spatialmamba_s all
#
# Add --dry-run as a third argument to print what would launch, and exit.
# ==========================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMDET_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${MMDET_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CFG_DIR="configs/Custom/maskrcnn_palm_staged"
STAGE_C_ROOT="/workspace/mmdetection/work_dirs/Stage_C"
SIM_ROOT="/workspace/mmdetection/work_dirs/Stage_D/Simulated"
WORK_ROOT="/workspace/mmdetection/work_dirs/Stage_D"

BACKBONES=(convnext_t swin_s spatialmamba_s mambavision_s)

DRY_RUN=0

# --- resolve a single checkpoint from a glob, or die ----------------------
resolve_one() {
  local label="$1"; shift
  local matches=()
  local g
  for g in "$@"; do
    # shellcheck disable=SC2206
    local hit=( $g )
    [[ -e "${hit[0]:-}" ]] && matches+=("${hit[@]}")
  done
  if (( ${#matches[@]} == 0 )); then
    echo "ERROR: no checkpoint found for ${label}. Looked for: $*" >&2
    return 1
  fi
  if (( ${#matches[@]} > 1 )); then
    echo "ERROR: ambiguous checkpoint for ${label} (${#matches[@]} matches):" >&2
    printf '   %s\n' "${matches[@]}" >&2
    return 1
  fi
  echo "${matches[0]}"
}

stagec_ckpt() {   # $1 = backbone
  resolve_one "Stage C best_GE (${1})" \
    "${STAGE_C_ROOT}/maskrcnn_${1}_stagec/best_GE_segm_mAP_50_iter_*.pth"
}

ge30sim_ckpt() {  # $1 = backbone
  # The Simulated/ directories do NOT follow the backbone names used
  # everywhere else: three carry the size suffix (convnext_t, swin_s,
  # mambavision_s) and Spatial-Mamba does not (spatialmamba, not
  # spatialmamba_s). Both spellings are offered; resolve_one dies on
  # ambiguity, so a future rename that makes both exist is caught rather
  # than silently resolving to whichever the glob returned first.
  local globs=("${SIM_ROOT}/${1}_ge30sim_stage1/best_*segm_mAP_50_iter_*.pth")
  # Only add the stripped spelling when it is genuinely different. For
  # convnext_t the suffix strip is a no-op, and offering the same glob twice
  # would match one file twice and be reported as ambiguous.
  if [[ "${1%_s}" != "${1}" ]]; then
    globs+=("${SIM_ROOT}/${1%_s}_ge30sim_stage1/best_*segm_mAP_50_iter_*.pth")
  fi
  resolve_one "GE-30sim stage 1 (${1})" "${globs[@]}"
}

launch() {        # $1 = backbone  $2 = arm(b0|c|cf|s)
  local bb="$1" arm="$2"
  local cfg wd
  local opts=()

  case "${arm}" in
    b0)
      # Full training. Its own config, and deliberately NO load_from: the
      # backbone init_cfg supplies ImageNet. Passing one here would turn
      # arm b0 into arm c without changing anything visible in the log.
      cfg="${CFG_DIR}/maskrcnn_${bb}_staged_full.py"
      wd="${WORK_ROOT}/maskrcnn_${bb}_staged_full"
      ;;
    c)
      cfg="${CFG_DIR}/maskrcnn_${bb}_staged_ft.py"
      wd="${WORK_ROOT}/maskrcnn_${bb}_staged_ft/c"
      opts+=("load_from=$(stagec_ckpt "${bb}")")
      ;;
    cf)
      # The b0 config -- same schedule, same freezing, same samplers -- with
      # the Stage C prior injected. No separate config file: any difference
      # between this and b0 beyond load_from would defeat the purpose.
      cfg="${CFG_DIR}/maskrcnn_${bb}_staged_full.py"
      wd="${WORK_ROOT}/maskrcnn_${bb}_staged_cfull"
      opts+=("load_from=$(stagec_ckpt "${bb}")")
      ;;
    s)
      # Same config as b0 and cf. Only load_from differs -- that is the
      # entire point of the arm.
      cfg="${CFG_DIR}/maskrcnn_${bb}_staged_full.py"
      wd="${WORK_ROOT}/maskrcnn_${bb}_staged_sfull"
      opts+=("load_from=$(ge30sim_ckpt "${bb}")")
      ;;
    bu)
      echo "ERROR: arm 'bu' is cut from Stage D (companion paper)." >&2
      echo "       See ${CFG_DIR}/STAGE_D_README.md section 4." >&2
      return 1
      ;;
    *)
      echo "ERROR: unknown arm '${arm}' (expected b0, c, cf or s)" >&2
      return 1
      ;;
  esac

  [[ -f "${cfg}" ]] || { echo "ERROR: missing config ${cfg}" >&2; return 1; }

  # Skip anything already finished. Without this, re-running after a
  # cancellation repeats completed models from scratch -- six hours each, and
  # the second result would differ from the first only by seed noise while
  # overwriting the checkpoints the table was built from.
  #
  # COMPLETION IS A SENTINEL, NOT A CHECKPOINT. An earlier version skipped any
  # work_dir holding a best_*.pth, which is wrong: save_best writes one at the
  # FIRST validation, so a run killed at iteration 2,400 of 60,000 looks
  # identical to one that trained to convergence. That happened -- a MambaVision
  # run died early and would silently have been skipped, leaving a hole in the
  # matrix that only showed up as a missing row much later. The sentinel is
  # written after tools/train.py exits 0 and means nothing else.
  if [[ -f "${wd}/.run_complete" ]]; then
    echo "=== ${bb} arm=${arm}: SKIP, already complete"
    [[ -n "$(compgen -G "${wd}/best_coco_segm_mAP_50_iter_*.pth" || true)" ]] && \
      echo "    $(basename "$(ls -1 "${wd}"/best_coco_segm_mAP_50_iter_*.pth | head -1)")"
    echo "    delete ${wd} to force a re-run"
    return 0
  fi
  # A checkpoint without the sentinel is an interrupted run. Say so loudly --
  # it is about to be restarted from scratch and its old checkpoints
  # overwritten, which is right, but not something to discover afterwards.
  if compgen -G "${wd}/best_coco_segm_mAP_50_iter_*.pth" > /dev/null; then
    echo "=== ${bb} arm=${arm}: WARNING, checkpoints present but no .run_complete"
    echo "    previous attempt did not finish; restarting from scratch"
  fi

  echo "=== ${bb} arm=${arm}"
  echo "    config   : ${cfg}"
  echo "    work_dir : ${wd}"
  if (( ${#opts[@]} > 0 )); then
    printf '    cfg-option: %s\n' "${opts[@]}"
  else
    echo "    cfg-option: (none -- ImageNet via backbone init_cfg)"
  fi
  (( DRY_RUN )) && return 0

  mkdir -p "${wd}"
  rm -f "${wd}/.run_complete"
  if (( ${#opts[@]} > 0 )); then
    python tools/train.py "${cfg}" --work-dir "${wd}" \
        --cfg-options "${opts[@]}" 2>&1 | tee "${wd}/launch.log"
  else
    python tools/train.py "${cfg}" --work-dir "${wd}" \
        2>&1 | tee "${wd}/launch.log"
  fi
  # PIPESTATUS[0] is train.py's exit code; $? would be tee's, which is 0 even
  # when training crashed.
  local rc=${PIPESTATUS[0]}
  if (( rc == 0 )); then
    touch "${wd}/.run_complete"
  else
    echo "=== ${bb} arm=${arm}: FAILED (exit ${rc}); no completion sentinel" >&2
    return "${rc}"
  fi
}

# --- dispatch -------------------------------------------------------------
WHICH_BB="${1:?usage: run_staged_matrix.sh <backbone|all> <b0|c|cf|s|all> [--dry-run]}"
ARM="${2:?missing arm (b0|c|cf|s|all)}"
[[ "${3:-}" == "--dry-run" ]] && DRY_RUN=1

if [[ "${WHICH_BB}" == "all" ]]; then
  TARGETS=("${BACKBONES[@]}")
else
  TARGETS=("${WHICH_BB}")
fi

if [[ "${ARM}" == "all" ]]; then
  # The three matched-recipe initialisations: ImageNet, Stage C, GE-30sim.
  # Identical config throughout, so the only variable is where the weights
  # came from. Arm c is available explicitly but is not part of the default
  # matrix -- it changes the recipe as well, so it cannot be read against
  # these -- see the header.
  ARMS=(b0 cf s)
else
  ARMS=("${ARM}")
fi

mkdir -p "${WORK_ROOT}"
for bb in "${TARGETS[@]}"; do
  for a in "${ARMS[@]}"; do
    launch "${bb}" "${a}"
  done
done
