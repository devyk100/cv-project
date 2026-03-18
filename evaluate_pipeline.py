import argparse
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from detection.edge_detection import EdgeDetector, WeightedFasterRCNN
from enhancement.cloud_enhancement import CloudEnhancer
from evaluation.exdark_eval import CLASS_MAP, evaluate_dataset
from preprocessing.single_preprocess import preprocess_single_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# COCO label ids for torchvision Faster R-CNN
_COCO_NAME_TO_ID = {
    "person": 1,
    "bicycle": 2,
    "car": 3,
    "motorcycle": 4,
    "bus": 6,
    "boat": 9,
    "cat": 17,
    "dog": 18,
    "bottle": 44,
    "cup": 47,
    "chair": 62,
    "dining table": 67,
}

# Map COCO label id -> ExDark class index
COCO_TO_EXDARK = {
    _COCO_NAME_TO_ID["bicycle"]: CLASS_MAP["Bicycle"],
    _COCO_NAME_TO_ID["boat"]: CLASS_MAP["Boat"],
    _COCO_NAME_TO_ID["bottle"]: CLASS_MAP["Bottle"],
    _COCO_NAME_TO_ID["bus"]: CLASS_MAP["Bus"],
    _COCO_NAME_TO_ID["car"]: CLASS_MAP["Car"],
    _COCO_NAME_TO_ID["cat"]: CLASS_MAP["Cat"],
    _COCO_NAME_TO_ID["chair"]: CLASS_MAP["Chair"],
    _COCO_NAME_TO_ID["cup"]: CLASS_MAP["Cup"],
    _COCO_NAME_TO_ID["dog"]: CLASS_MAP["Dog"],
    _COCO_NAME_TO_ID["motorcycle"]: CLASS_MAP["Motorbike"],
    _COCO_NAME_TO_ID["person"]: CLASS_MAP["People"],
    _COCO_NAME_TO_ID["dining table"]: CLASS_MAP["Table"],
}


def _list_images(image_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    names = [n for n in os.listdir(image_dir) if n.lower().endswith(exts)]
    names.sort()
    return names


def _build_annotation_keys(annotation_root: str) -> set[str]:
    keys: set[str] = set()
    for root, _, files in os.walk(annotation_root):
        for fn in files:
            if not fn.endswith(".txt"):
                continue
            keys.add(fn[:-4])  # strip ".txt"
    return keys


def _normalize_image_key(img_filename: str, ann_keys: set[str]) -> str:
    """
    Map an image filename from --image-dir to the key expected by ExDark annotations.

    Common case in this repo: evaluation images are named "Y_2015_00001.png"
    while annotations are keyed like "2015_00001.jpg" (so annotation file is
    "2015_00001.jpg.txt").
    """
    base = img_filename
    if base.startswith("Y_"):
        base = base[2:]

    if base in ann_keys:
        return base

    stem, ext = os.path.splitext(base)
    # Try common swaps.
    for candidate in (stem + ".jpg", stem + ".png", stem + ".jpeg"):
        if candidate in ann_keys:
            return candidate

    # Fall back to base (will raise a clear annotation-not-found error later).
    return base


def _to_exdark_predictions(
    detections: List[Dict[str, torch.Tensor]],
    orig_hw: Tuple[int, int],
    detector_size: int = 800,
    score_threshold: float = 0.05,
    max_dets: int = 300,
) -> List[Dict]:
    """
    Convert torchvision Faster R-CNN outputs into ExDark evaluator format.

    Output format:
        [{"bbox":[x1,y1,x2,y2], "score":float, "class":int}, ...]
    with bbox coordinates in ORIGINAL image pixels (xyxy).
    """
    if not detections:
        return []

    det0 = detections[0]
    if "boxes" not in det0 or det0["boxes"].numel() == 0:
        return []

    boxes = det0["boxes"].detach().cpu().numpy()
    labels = det0["labels"].detach().cpu().numpy()
    scores = det0["scores"].detach().cpu().numpy()

    # Filter by score first (keeps ordering by model confidence).
    keep = scores >= float(score_threshold)
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    if boxes.shape[0] == 0:
        return []

    # Keep only ExDark-relevant classes and map labels.
    mapped = []
    for b, l, s in zip(boxes, labels, scores):
        ex_cls = COCO_TO_EXDARK.get(int(l))
        if ex_cls is None:
            continue
        mapped.append((b, float(s), int(ex_cls)))

    if not mapped:
        return []

    # Enforce max detections.
    mapped = mapped[: int(max_dets)]

    h, w = orig_hw
    scale_x = w / float(detector_size)
    scale_y = h / float(detector_size)

    out: List[Dict] = []
    for b, s, ex_cls in mapped:
        x1, y1, x2, y2 = b.tolist()
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        # Clamp to image bounds.
        x1 = float(np.clip(x1, 0, w - 1))
        x2 = float(np.clip(x2, 0, w - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        y2 = float(np.clip(y2, 0, h - 1))

        out.append({"bbox": [x1, y1, x2, y2], "score": s, "class": ex_cls})

    return out


def run_baseline_on_image(
    edge: EdgeDetector,
    image_bgr: np.ndarray,
    score_threshold: float,
    max_dets: int,
) -> List[Dict]:
    h, w = image_bgr.shape[:2]
    detector_input = cv2.resize(image_bgr, (800, 800))
    tensor = torch.tensor(detector_input).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        det = edge.model(tensor)
    return _to_exdark_predictions(det, (h, w), score_threshold=score_threshold, max_dets=max_dets)


def run_enhanced_on_image(
    cloud: CloudEnhancer,
    edge: EdgeDetector,
    detector: WeightedFasterRCNN,
    image_path: str,
    orig_hw: Tuple[int, int],
    score_threshold: float,
    max_dets: int,
) -> List[Dict]:
    # Enhancement path operates on a 224x224 version of the image.
    rgb_resized, Y, Cr, Cb = preprocess_single_image(image_path)
    Y_tensor = torch.tensor(Y)

    enhanced_images, weights = cloud.enhance_tensor(Y_tensor)
    rgb_images = edge.reconstruct_rgb(enhanced_images, Cr, Cb)

    feats = edge.extract_features(rgb_images)
    orig_feat = edge.extract_features([rgb_resized])[0]
    weighted = edge.apply_weights(feats, weights)
    fused = edge.combine_features(orig_feat, weighted)

    detector_input = cv2.resize(rgb_resized, (800, 800))
    tensor = torch.tensor(detector_input).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        det = detector.detect_with_features(tensor, fused)

    return _to_exdark_predictions(det, orig_hw, score_threshold=score_threshold, max_dets=max_dets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs edge-cloud enhanced pipeline on ExDark-style annotations.")
    parser.add_argument("--image-dir", default="enhancement/dataset/input", help="Directory containing images to evaluate.")
    parser.add_argument("--annotation-root", default=None, help="Root directory containing ExDark bbGt .txt annotations.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "enhanced", "both"],
        default="both",
        help="Which pipeline(s) to run.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.05, help="Minimum score to keep a detection.")
    parser.add_argument("--max-dets", type=int, default=300, help="Max detections per image (after filtering/mapping).")
    parser.add_argument("--save-json", default=None, help="Optional path to save predictions/results JSON.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "enhancement/model_clahe.pth",
            "enhancement/model_hist.pth",
            "enhancement/model_sharpen.pth",
            "enhancement/model_bilateral.pth",
            "enhancement/model_gamma.pth",
        ],
        help="Enhancement model checkpoints (only used in enhanced/both modes).",
    )
    parser.add_argument("--no-eval", action="store_true", help="Only run inference and dump predictions (skip metric computation).")
    args = parser.parse_args()

    image_dir = args.image_dir
    if not os.path.isdir(image_dir):
        raise SystemExit(f"Image directory not found: {image_dir}")

    images = _list_images(image_dir)
    if not images:
        raise SystemExit(f"No images found in {image_dir}")

    ann_keys: set[str] = set()
    if (not args.no_eval) and args.annotation_root:
        ann_keys = _build_annotation_keys(args.annotation_root)

    edge = EdgeDetector()

    cloud = None
    weighted_detector = None
    if args.mode in ("enhanced", "both"):
        cloud = CloudEnhancer(args.models)
        weighted_detector = WeightedFasterRCNN()

    baseline_preds: Dict[str, List[Dict]] = {}
    enhanced_preds: Dict[str, List[Dict]] = {}

    eval_image_keys: List[str] = []

    for name in images:
        path = os.path.join(image_dir, name)
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        key = _normalize_image_key(name, ann_keys) if ann_keys else (name[2:] if name.startswith("Y_") else name)
        eval_image_keys.append(key)
        if args.mode in ("baseline", "both"):
            baseline_preds[key] = run_baseline_on_image(edge, img, args.score_threshold, args.max_dets)
        if args.mode in ("enhanced", "both"):
            assert cloud is not None and weighted_detector is not None
            enhanced_preds[key] = run_enhanced_on_image(
                cloud,
                edge,
                weighted_detector,
                path,
                (h, w),
                args.score_threshold,
                args.max_dets,
            )

    output: Dict[str, object] = {"image_dir": image_dir, "num_images": len(images), "mode": args.mode}

    if args.no_eval:
        if args.mode in ("baseline", "both"):
            output["baseline_predictions"] = baseline_preds
        if args.mode in ("enhanced", "both"):
            output["enhanced_predictions"] = enhanced_preds
    else:
        if not args.annotation_root:
            raise SystemExit("--annotation-root is required unless --no-eval is set")
        ann_root = args.annotation_root
        if not os.path.isdir(ann_root):
            raise SystemExit(f"Annotation root not found: {ann_root}")

        if args.mode in ("baseline", "both"):
            output["baseline_results"] = evaluate_dataset(eval_image_keys, ann_root, baseline_preds)
        if args.mode in ("enhanced", "both"):
            output["enhanced_results"] = evaluate_dataset(eval_image_keys, ann_root, enhanced_preds)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()