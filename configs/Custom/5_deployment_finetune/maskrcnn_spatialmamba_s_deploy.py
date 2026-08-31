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
#
# WHY 2000 FOR THE SECOND NATIONAL RUN
#   The cap applies to the model's own output at its internal score_thr of
#   0.05, so lowering the PIPELINE threshold does not change when it binds --
#   but it does change what the binding costs. Once a tile holds more than
#   max_per_img detections above 0.05, the surplus is discarded by score, and
#   at a pipeline threshold of 0.15 more of that surplus would have survived
#   than at 0.30.
#
#   The headroom is thinner than it looks. A 1024 px tile at 14.9 cm covers
#   153 x 153 m = 2.34 ha; date palms at 5 x 5 m spacing put roughly 936
#   crowns in one tile before any sub-threshold detection is counted. Raising
#   the cap was measured at 0.30 as -0.036% on UAE_60 -- noise -- so this is
#   insurance rather than a correction, and it cannot remove a detection:
#   ownership, the shape gate and polygon NMS all still apply downstream.
# ==========================================================================

_base_ = ['../3_unified_multisource/maskrcnn_spatialmamba_s_stagec.py']

model = dict(
    test_cfg=dict(
        rcnn=dict(max_per_img=2000)))
