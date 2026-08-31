# 5 — Deployment and hard-negative adaptation

*Internal name: the `finetune_hn` set. Inherits the unified model from
`3_unified_multisource/`.*

**The question.** A benchmark model meets a whole country. What breaks?

Terrain the training corpus never contained as labelled background:
palm-like shrubs, ghaf, acacia. The model scores them as palms. Nothing in
the benchmark measures this, because the benchmark's test tiles contain
palms.

**This is an operational adaptation, not part of the benchmark.** The
unified checkpoints and their reported numbers are untouched. Describe it
as a post-processing step, not as a competing model.

## How it works

Mask R-CNN trains on empty images natively: on a tile with no annotations,
every proposal is an unmatched negative, so the tile is pure "nothing here
is a palm" supervision. The benchmark sets `filter_empty_gt=True`, which
*skips* such tiles — a choice, not a framework limit.

This set turns it off and adds tiles containing the exact confusers the
model fails on.

## The guard that matters

Fine-tuning on negatives alone erodes recall — the model learns to say no.
The original positive training data is **replayed alongside** the
negatives.

**If you build a variant of this, keep the replay.** Removing it is the
easiest way to produce a model that reports far fewer false positives and
quietly finds fewer palms.

## The configs

| config | what it is |
|---|---|
| `maskrcnn_spatialmamba_s_finetune_hn.py` | the hard-negative adaptation |
| `maskrcnn_spatialmamba_s_finetune_fn.py` | the false-negative counterpart |
| `maskrcnn_spatialmamba_s_deploy.py` | **the deployed configuration** |

The deploy config raises the per-image detection cap. Its header explains
why: the cap binds only in dense plantations, validation tiles were never
that dense, and so validation could not have revealed the problem.

## After training, two things are not optional

```bash
python configs/Custom/Finetune_HN/calibrate_threshold.py   # re-derive the threshold
python configs/Custom/Finetune_HN/eval_hard_negatives.py   # measure BOTH axes
```

A fine-tuned model does not inherit the old operating point. And standard
COCO mAP is close to blind to false-positive suppression — a tile with no
ground truth contributes no true positives — so measure it directly.

**Check both axes.** Suppressing false positives while losing recall has
not improved anything; it has moved the failure somewhere less visible.

## Running the deployed model

See the repository README, "Running the deployed model". Note that the
inference CLI defaults are **not** the deployment settings.
