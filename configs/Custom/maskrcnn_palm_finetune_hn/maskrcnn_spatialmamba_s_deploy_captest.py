# ==========================================================================
# maskrcnn_spatialmamba_s_deploy.py
# --------------------------------------------------------------------------
# The DEPLOYMENT config for the country-scale inventory. Architecture is
# identical to Stage C Spatial-Mamba-S; only the inference-time detection cap
# differs.
#
# WHY THIS FILE EXISTS INSTEAD OF AN EDIT TO THE SHARED BASE
#   max_per_img lives in _base_palm/_base_maskrcnn_palm_stagec.py, which every
#   one of the ten Stage C benchmark configs inherits. Raising it there would
#   silently change the benchmark those configs produced and invalidate the
#   published comparison. The deployment run needs a different cap from the
#   benchmark, so it gets its own config and the benchmark is left alone.
#
# WHY 1000 AND NOT 300
#   The cap is not a threshold. Every detection still has to clear the score
#   threshold and NMS first; the cap only discards the surplus once a tile
#   already holds more than max_per_img survivors. On sparse terrain it never
#   binds, so it cannot add false positives.
#
#   It binds only in dense plantations, and there it silently truncates. GE
#   validation measures the effect: average recall is 0.577 at 100 detections,
#   0.655 at 300, and 0.655 at 1000. Flat from 300 to 1000 means no validation
#   tile is dense enough for the old cap to bite -- which is precisely why the
#   old cap looked safe. Country-scale plantations are denser than any
#   validation tile, so the risk is real and unmeasured there, while the cost
#   of raising it is zero everywhere the validation could see.
# ==========================================================================

_base_ = ['../maskrcnn_palm_stagec/maskrcnn_spatialmamba_s_stagec.py']

model = dict(
    test_cfg=dict(
        rcnn=dict(max_per_img=5000)))
