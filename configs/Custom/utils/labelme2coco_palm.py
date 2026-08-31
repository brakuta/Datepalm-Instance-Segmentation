#!/usr/bin/env python3
# =============================================================================
# labelme2coco_palm.py
# -----------------------------------------------------------------------------
# LabelMe -> COCO converter for the palm tiling pipeline.
#
# Derived from labelme's own examples/instance_segmentation/labelme2coco.py,
# with four changes that the stock script gets wrong for this project.
#
# 1. iscrowd FOLLOWS THE PARTIAL FLAG
#    The stock converter hardcodes iscrowd=0. image_vector_to_labelme_pipeline
#    marks crowns cut by a tile edge with flags={"partial": true}; those must
#    become iscrowd=1 so COCO ignores the region instead of scoring it. With
#    iscrowd=0 a 20%-visible crown sliver is taught as a complete palm, which
#    is how you get double counting at tile seams.
#
#    THIS ONLY WORKS IF THE MODEL HONOURS IGNORE REGIONS. MMDetection's
#    MaxIoUAssigner uses them only when ignore_iof_thr > 0; the project base
#    sets -1, which silently discards them and puts the crown pixels back into
#    the background. The Stage D configs override it to 0.5. If you convert a
#    dataset for a config that does not, use PARTIAL_POLICY="drop" upstream
#    instead and accept the false negatives knowingly.
#
# 2. DETERMINISTIC image_id
#    The stock script enumerates glob.glob(), whose order is filesystem order.
#    Two conversions of the same directory could assign different image_ids,
#    which makes a results .pkl or an error analysis non-portable between
#    machines. Sorted here.
#
# 3. NO labelme / imgviz DEPENDENCY
#    Masks are built with pycocotools.frPyObjects -- the same rasterisation
#    COCOeval uses -- rather than labelme's PIL path, so area and bbox agree
#    exactly with what the metric will later compute. Pixels are read with
#    rasterio when available, which handles multi-band GeoTIFF properly.
#
# 4. EMPTY TILES SURVIVE
#    Tiles with no annotations are still written to "images". They are
#    deliberate background supervision.
#
#    MMDetection will THROW THEM AWAY AGAIN unless the train dataloader sets
#      filter_cfg=dict(filter_empty_gt=False, ...)
#    This script prints a reminder whenever it emits any. Check the training
#    log's dataset line: if the image count is lower than reported here, the
#    filter is still on and the background tiles never reached the model.
#
# USAGE
#   python configs/Custom/utils/labelme2coco_palm.py \
#       /path/to/tiles/train/images \
#       /workspace/datasets/COCO/Sat_30cm/train_sat \
#       --labels configs/Custom/utils/labels.txt
# =============================================================================

from __future__ import annotations

import argparse
import collections
import datetime
import json
import sys
from pathlib import Path

import numpy as np

try:
    import pycocotools.mask as maskUtils
except ImportError:
    sys.exit("Please install pycocotools:\n\n    pip install pycocotools\n")


def load_image(path: Path) -> np.ndarray:
    """Read a tile as HxWxC uint8. rasterio for GeoTIFF, PIL otherwise."""
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import rasterio
            with rasterio.open(path) as src:
                arr = src.read()                      # (C, H, W)
            return np.transpose(arr, (1, 2, 0))
        except ImportError:
            pass
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="directory of LabelMe .json + image pairs")
    ap.add_argument("output_dir", nargs="?", default=None,
                    help="ad-hoc mode: a self-contained COCO directory. "
                         "Prefer --dataset-root/--split-name, which writes "
                         "straight into the project layout.")
    ap.add_argument("--dataset-root", default=None,
                    help="COCO root, e.g. /workspace/datasets/COCO/Sat_30cm")
    ap.add_argument("--split-name", default=None,
                    help="split directory and annotation stem, e.g. train_sat. "
                         "Images -> <root>/<split-name>/JPEGImages/, "
                         "annotations -> <root>/Annotations/<split-name>.json, "
                         "which is exactly what the dataset configs read "
                         "(data_prefix img='<split-name>/', ann_file="
                         "'Annotations/<split-name>.json').")
    ap.add_argument("--labels", required=True,
                    help="labels file; first line must be __ignore__")
    ap.add_argument("--image-format", choices=("jpg", "tif"), default="jpg",
                    help="jpg matches the existing JPEGImages convention of "
                         "the other splits and is limited to 3 bands. tif "
                         "passes ALL bands through with no recompression and "
                         "is the only option for multispectral. Do not mix the "
                         "two within one experiment.")
    ap.add_argument("--rgb-bands", nargs=3, type=int, default=None,
                    metavar=("R", "G", "B"),
                    help="1-based bands to write when --image-format jpg and "
                         "the source has more than three. Required in that "
                         "case: WorldView-3 8-band order is Coastal, Blue, "
                         "Green, Yellow, Red, RedEdge, NIR1, NIR2, so the "
                         "first three bands are NOT RGB. True colour is "
                         "5 3 2; NIR-R-G false colour is 7 5 3.")
    ap.add_argument("--quality", type=int, default=95,
                    help="JPEG quality when --image-format jpg")
    ap.add_argument("--force", action="store_true",
                    help="write into an existing output directory")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)

    # Two layouts. The project one puts images and annotations in the places
    # the dataset configs already look, so nothing has to be moved by hand
    # afterwards -- the step where a split most easily gets mixed up.
    if args.dataset_root or args.split_name:
        if not (args.dataset_root and args.split_name):
            sys.exit("--dataset-root and --split-name must be given together.")
        root     = Path(args.dataset_root)
        out_dir  = root / args.split_name
        img_out  = out_dir / "JPEGImages"
        out_ann  = root / "Annotations" / f"{args.split_name}.json"
        out_ann.parent.mkdir(parents=True, exist_ok=True)
    else:
        if not args.output_dir:
            sys.exit("Give either output_dir, or --dataset-root with "
                     "--split-name.")
        out_dir = Path(args.output_dir)
        img_out = out_dir / "JPEGImages"
        out_ann = out_dir / "annotations.json"

    if out_ann.exists() and not args.force:
        sys.exit(f"Annotation file already exists: {out_ann}\n"
                 f"Pass --force to overwrite.")
    img_out.mkdir(parents=True, exist_ok=True)

    # ---- categories --------------------------------------------------------
    class_name_to_id = {}
    categories = []
    for i, line in enumerate(Path(args.labels).read_text().splitlines()):
        class_id, class_name = i - 1, line.strip()
        if class_id == -1:
            if class_name != "__ignore__":
                sys.exit(f"First line of --labels must be __ignore__, "
                         f"got {class_name!r}")
            continue
        if not class_name:
            continue
        class_name_to_id[class_name] = class_id
        categories.append(dict(supercategory=None, id=class_id,
                               name=class_name))

    now = datetime.datetime.now()
    data = dict(
        info=dict(description=None, url=None, version=None, year=now.year,
                  contributor=None,
                  date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f")),
        licenses=[dict(url=None, id=0, name=None)],
        images=[], type="instances", annotations=[], categories=categories,
    )

    # Sorted, not glob order: image_id must be reproducible.
    label_files = sorted(in_dir.glob("*.json"))
    if not label_files:
        sys.exit(f"No .json files under {in_dir}")

    n_empty = n_crowd = n_missing = 0
    band_counts = set()

    for image_id, jp in enumerate(label_files):
        d = json.loads(jp.read_text(encoding="utf-8"))

        src_img = jp.parent / d.get("imagePath", jp.stem + ".tif")
        if not src_img.exists():
            n_missing += 1
            print(f"  [warn] missing image for {jp.name}: {src_img.name}")
            continue
        img = load_image(src_img)
        h, w = img.shape[:2]

        n_band = 1 if img.ndim == 2 else img.shape[2]
        band_counts.add(n_band)

        if args.image_format == "jpg":
            from PIL import Image
            if n_band > 3 and args.rgb_bands is None:
                sys.exit(
                    f"{src_img.name} has {n_band} bands and --image-format is "
                    f"jpg, which holds 3.\nTaking the first three silently "
                    f"would be wrong for most multispectral products -- "
                    f"WorldView-3 band 1-3 is Coastal/Blue/Green, not RGB.\n"
                    f"Either state the composite, e.g. --rgb-bands 5 3 2 "
                    f"(true colour) or 7 5 3 (NIR-R-G),\nor keep every band "
                    f"with --image-format tif.")
            if args.rgb_bands:
                bad = [b for b in args.rgb_bands if not 1 <= b <= n_band]
                if bad:
                    sys.exit(f"--rgb-bands {bad} out of range for a "
                             f"{n_band}-band source ({src_img.name})")
                sel = img[:, :, [b - 1 for b in args.rgb_bands]]
            else:
                sel = img[:, :, :3] if n_band >= 3 else \
                    np.repeat(img.reshape(h, w, 1), 3, axis=2)
            dst = img_out / (jp.stem + ".jpg")
            Image.fromarray(sel.astype(np.uint8)).save(dst,
                                                       quality=args.quality)
        else:
            # Every band, byte for byte. This is the multispectral path.
            import shutil
            dst = img_out / src_img.name
            shutil.copy2(src_img, dst)

        data["images"].append(dict(
            license=0, url=None, file_name=str(Path("JPEGImages") / dst.name),
            height=h, width=w, date_captured=None, id=image_id))

        # ---- group shapes into instances -----------------------------------
        # One crown may arrive as several polygons (a tile corner can split it)
        # sharing a group_id; they are one COCO annotation with a multi-polygon
        # segmentation. An instance is crowd if ANY of its parts is flagged
        # partial -- a crown that is cut is cut, whichever piece says so.
        segs = collections.defaultdict(list)
        crowd = collections.defaultdict(bool)
        for shape in d.get("shapes", []):
            if shape.get("shape_type", "polygon") != "polygon":
                continue
            label = shape["label"]
            if label not in class_name_to_id:
                continue
            gid = shape.get("group_id")
            key = (label, gid if gid is not None else f"_solo{len(segs)}")
            pts = np.asarray(shape["points"], float).flatten().tolist()
            if len(pts) < 6:                     # fewer than 3 vertices
                continue
            segs[key].append(pts)
            if shape.get("flags", {}).get("partial"):
                crowd[key] = True

        if not segs:
            n_empty += 1

        for (label, _gid), polys in segs.items():
            # frPyObjects + merge is exactly the rasterisation COCOeval uses,
            # so area and bbox here match what the metric will recompute.
            rles = maskUtils.frPyObjects(polys, h, w)
            rle  = maskUtils.merge(rles)
            area = float(maskUtils.area(rle))
            if area <= 0:
                continue
            is_crowd = 1 if crowd[(label, _gid)] else 0
            n_crowd += is_crowd
            data["annotations"].append(dict(
                id=len(data["annotations"]), image_id=image_id,
                category_id=class_name_to_id[label],
                segmentation=polys, area=area,
                bbox=maskUtils.toBbox(rle).flatten().tolist(),
                iscrowd=is_crowd))

    out_ann.write_text(json.dumps(data))

    n_img, n_ann = len(data["images"]), len(data["annotations"])
    print(f"\nimages      : {n_img}")
    print(f"annotations : {n_ann}  ({n_ann - n_crowd} instances, "
          f"{n_crowd} iscrowd=1 ignore regions)")
    print(f"empty tiles : {n_empty}")
    print(f"bands       : {sorted(band_counts)} in source, "
          f"{'3 written (jpg)' if args.image_format == 'jpg' else 'all written (tif)'}"
          + (f", composite {args.rgb_bands}" if args.rgb_bands else ""))
    if n_missing:
        print(f"MISSING IMG : {n_missing} json(s) had no matching image and "
              f"were skipped")
    print(f"-> {out_ann}")

    if n_empty:
        print(f"\n{n_empty} tile(s) carry no annotations. They are background "
              f"supervision and MMDetection discards them by default.\n"
              f"The train dataloader must set "
              f"filter_cfg=dict(filter_empty_gt=False, ...) or they never "
              f"reach the model.\nConfirm against the image count in the "
              f"training log.")
    if args.image_format == "tif" and max(band_counts or [3]) > 3:
        print(f"\nMultispectral output: {max(band_counts)} bands per tile.\n"
              f"MMDetection will NOT read these correctly out of the box. The "
              f"training config needs, at minimum:\n"
              f"  * a loader that keeps every band (mmcv.imread returns 3-"
              f"channel BGR and would silently drop the rest)\n"
              f"  * data_preprocessor mean/std with {max(band_counts)} entries\n"
              f"  * a backbone stem accepting {max(band_counts)} input "
              f"channels -- ImageNet weights are 3-channel, so the extra "
              f"channels start untrained\n"
              f"See TILING_README.md, 'Multispectral'.")

    if n_crowd:
        print(f"\n{n_crowd} annotation(s) are iscrowd=1 (crowns cut by a tile "
              f"edge). These are ignored rather than scored ONLY where the\n"
              f"assigner sets ignore_iof_thr > 0. The project base uses -1; "
              f"the Stage D configs override it to 0.5.")


if __name__ == "__main__":
    main()
