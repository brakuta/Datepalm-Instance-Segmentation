# Experiment 5: deployment and hard-negative adaptation

*Internal name: the `finetune_hn` set. Inherits the unified model from
`3_unified_multisource/`.*

This experiment adapts the experiment 3 model for country-scale
operation. Run at that scale, the benchmark model meets terrain the
training corpus never contained as labelled background (palm-like
shrubs, ghaf, acacia) and scores it as palms. The benchmark does not
measure this failure mode, because the benchmark's test tiles contain
palms.

**This is an operational adaptation, not part of the benchmark.** The
unified checkpoints and their reported numbers are untouched. Describe it
as a post-processing step, not as a competing model.

## How it works

Mask R-CNN trains on empty images natively: on a tile with no
annotations, every proposal is an unmatched negative, so an empty tile
provides pure background supervision. The benchmark sets
`filter_empty_gt=True`, which skips such tiles; that is a configuration
choice, not a framework limit. This set turns the filter off and adds
tiles containing the exact confusers the model fails on.

## Positive replay

Fine-tuning on negatives alone erodes recall, because the model is only
ever taught to suppress detections. The original positive training data
is therefore replayed alongside the negatives.

**If you build a variant of this, keep the replay.** Removing it
produces a model that reports far fewer false positives while also
finding fewer palms, and the recall loss raises no error; it shows up
only if measured.

## Configs

| config | purpose |
|---|---|
| `maskrcnn_spatialmamba_s_finetune_hn.py` | the hard-negative adaptation |
| `maskrcnn_spatialmamba_s_finetune_fn.py` | the false-negative counterpart |
| `maskrcnn_spatialmamba_s_deploy.py` | the deployed configuration |
| `maskrcnn_spatialmamba_s_deploy_captest.py` | cap probe: `max_per_img` raised to 5000 to measure where the cap binds |

The deploy config raises the per-image detection cap. Its header
explains why: the cap binds only in dense plantations, validation tiles
were never that dense, and so validation could not have revealed the
problem.

## After training

Two steps are required after every fine-tuning run:

```bash
python configs/Custom/Finetune_HN/calibrate_threshold.py   # re-derive the threshold
python configs/Custom/Finetune_HN/eval_hard_negatives.py   # measure BOTH axes
```

1. Recalibrate the threshold: a fine-tuned model does not inherit the
   old operating point.
2. Evaluate false-positive suppression directly: standard COCO mAP is
   close to blind to it, because a tile with no ground truth contributes
   no true positives.

Check both axes. A run that suppresses false positives at the cost of
recall has not improved the model.

## Running the deployed model

See the repository README, section "Inference with the deployed model".
The CLI defaults are the deployment settings (tile 1024, overlap 256,
threshold 0.30).

Four operational notes:

1. `--score-thr` is the pipeline threshold. The model always runs at its
   own internal `score_thr` of 0.05; this value decides what reaches the
   output.
2. `--overlap` must be at least the largest expected crown diameter in
   pixels, or palms on tile boundaries are cut in half and counted twice.
3. `--batch-size 6` is sized for a 24 GB card at tile size 1024. Halve it
   if you hit CUDA out-of-memory; this changes throughput, not results.
4. An interrupted run resumes from `_manifest.parquet` in the output
   root. Delete the manifest (and the `_consolidated*.gpkg` next to it)
   only if you intend to redo every image; leaving it in place is what
   makes the run resumable.
