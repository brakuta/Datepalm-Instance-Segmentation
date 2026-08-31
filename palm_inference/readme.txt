Country-scale inference — the command that was actually run.

The config path below is the DEPLOYMENT config and it exists in this tree.
An earlier version of this file named configs/Custom/VMamba/... which has
never existed here; copying it produced a file-not-found that looks like a
broken installation rather than a stale example.

Fill the checkpoint path from docs/handover/FACTS.yml — it is deliberately
not hard-coded here, so that this file and the facts file cannot disagree.

Clear the resume manifest ONLY if you intend to redo every image. Leaving it
in place is what makes an interrupted run resumable.

    rm -f /workspace/datasets/palm_output/_manifest.parquet
    rm -f /workspace/datasets/palm_output/_consolidated*.gpkg

    python -m palm_inference.run_inference \
        --input-root /workspace/datasets/UAE_imagery/Ajman \
        --output-root /workspace/datasets/palm_output \
        --config-file configs/Custom/maskrcnn_palm_finetune_hn/maskrcnn_spatialmamba_s_deploy.py \
        --checkpoint <deployed checkpoint — see docs/handover/FACTS.yml> \
        --tile-size 1024 --overlap 256 \
        --batch-size 6 --num-workers 4 \
        --score-thr 0.30 \
        --postprocess \
        --log-file /workspace/datasets/palm_output/run.log

Notes that cost time if you do not know them:

  --score-thr is the PIPELINE threshold. The model always runs at its own
  internal score_thr of 0.05; this value decides what reaches the output.

  --overlap must be at least the largest expected crown diameter in pixels,
  or palms on tile boundaries are cut in half and counted twice.

  --batch-size 6 is sized for a 24 GB card at tile-size 1024. Halve it if
  you hit CUDA out-of-memory; it changes throughput, not results.
