from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
import cv2
import numpy as np


def load_sam(model_type: str, checkpoint_path: str, device: str, verbose: bool):
    start = time.time()
    if verbose:
        print(f"[INFO] Loading SAM model '{model_type}' from: {checkpoint_path}", flush=True)
    try:
        device_to_use = torch.device(device if device in {"cuda", "cpu"} else ("cuda" if torch.cuda.is_available() else "cpu"))
        if verbose:
            print(f"[INFO] Using device: {device_to_use}", flush=True)
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device_to_use)
        if verbose:
            print(f"[OK] SAM loaded in {time.time()-start:.2f}s", flush=True)
        return sam, device_to_use
    except Exception as e:
        print(f"[ERROR] Failed to load SAM: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


def build_mask_generator(sam, points_per_side: int, pred_iou_thresh: float, stability_score_thresh: float,
                         crop_n_layers: int, min_mask_region_area: int):
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        min_mask_region_area=min_mask_region_area,
    )


def draw_preview(image_bgr: np.ndarray, masks: list) -> np.ndarray:
    preview = image_bgr.copy()
    for idx, m in enumerate(masks):
        x, y, w, h = m.get("bbox", [0, 0, 0, 0])
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(preview, f"id:{idx}", (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return preview


def save_mask_png(mask_bool: np.ndarray, out_path: str):
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255
    cv2.imwrite(out_path, mask_uint8)


def save_cropped_object(image_bgr: np.ndarray, mask_bool: np.ndarray, out_path: str):
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    cropped = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_uint8)
    cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, cropped_bgr)


def mask_to_polygons(mask_bool: np.ndarray):
    # Convert boolean mask to contours (COCO-style polygons)
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) >= 3:
            poly = cnt.reshape(-1, 2).astype(float).tolist()
            # Flatten for COCO-style segmentation (list of x,y)
            flat = [coord for point in poly for coord in point]
            polygons.append(flat)
    return polygons


def write_annotation_json(json_path: str, image_path: str, mask_id: int, mask_info: dict, label: str | None,
                          image_size: tuple[int, int]):
    h, w = image_size
    annotation = {
        "image_path": os.path.abspath(image_path),
        "image_width": int(w),
        "image_height": int(h),
        "selected_mask_id": int(mask_id),
        "label": label or "",
        "bbox": [int(v) for v in mask_info.get("bbox", [0, 0, 0, 0])],
        "area": int(mask_info.get("area", 0)),
        "predicted_iou": float(mask_info.get("predicted_iou", 0.0)),
        "stability_score": float(mask_info.get("stability_score", 0.0)),
        "polygons": mask_info.get("polygons", []),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(annotation, f, ensure_ascii=False, indent=2)


def list_images(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        images.extend([str(x) for x in p.glob(ext)])
    return sorted(images)


def select_mask_id(masks: list, strategy: str | None, provided_id: int | None) -> int | None:
    if provided_id is not None:
        return provided_id if 0 <= provided_id < len(masks) else None
    if strategy == "largest":
        if not masks:
            return None
        areas = [m.get("area", 0) for m in masks]
        return int(np.argmax(areas))
    return None


def process_image(image_path: str, mask_generator: SamAutomaticMaskGenerator, out_dir: str,
                  select_id: int | None, select_strategy: str | None, label: str | None, show: bool, verbose: bool):
    os.makedirs(out_dir, exist_ok=True)
    if verbose:
        print(f"[INFO] Reading image: {image_path}", flush=True)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"[WARN] Could not read image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if verbose:
        print(f"[INFO] Generating masks...", flush=True)
    gen_start = time.time()
    try:
        masks = mask_generator.generate(image_rgb)
    except Exception as e:
        print(f"[ERROR] Mask generation failed: {e}", flush=True)
        traceback.print_exc()
        return
    if verbose:
        print(f"[OK] Generated {len(masks)} masks in {time.time()-gen_start:.2f}s", flush=True)

    # Save preview with all IDs
    preview = draw_preview(image_bgr, masks)
    base = Path(image_path).stem
    preview_path = os.path.join(out_dir, f"{base}_preview.jpg")
    cv2.imwrite(preview_path, preview)
    if verbose:
        print(f"[OK] Preview saved: {preview_path}", flush=True)

    if show:
        try:
            cv2.imshow("SAM Preview", preview)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception:
            pass

    chosen_id = select_mask_id(masks, select_strategy, select_id)
    if chosen_id is None:
        print(f"[INFO] Preview saved at: {preview_path}. No selection made. Use --select-id or --select-largest.")
        return

    chosen = masks[chosen_id]
    mask_bool = chosen.get("segmentation").astype(bool)

    # Save mask PNG and cropped object
    mask_path = os.path.join(out_dir, f"{base}_mask_{chosen_id}.png")
    crop_path = os.path.join(out_dir, f"{base}_crop_{chosen_id}.png")
    save_mask_png(mask_bool, mask_path)
    save_cropped_object(image_bgr, mask_bool, crop_path)
    if verbose:
        print(f"[OK] Saved mask: {mask_path}", flush=True)
        print(f"[OK] Saved crop: {crop_path}", flush=True)

    # Enrich with polygons for JSON
    polygons = mask_to_polygons(mask_bool)
    chosen["polygons"] = polygons

    # Write JSON annotation
    json_path = os.path.join(out_dir, f"{base}_ann_{chosen_id}.json")
    h, w = image_bgr.shape[:2]
    write_annotation_json(json_path, image_path, chosen_id, chosen, label, (h, w))

    print(f"[OK] Saved preview: {preview_path}")
    print(f"[OK] Saved mask PNG: {mask_path}")
    print(f"[OK] Saved cropped object: {crop_path}")
    print(f"[OK] Saved annotation JSON: {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Segment Anything - Automatic mask selection and annotation utility")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM model checkpoint, e.g., vit_h.pth")
    parser.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"], help="SAM model type")
    parser.add_argument("--input", required=True, help="Path to an image file or a directory of images")
    parser.add_argument("--output", default="output", help="Directory to save previews, masks, crops, and JSON")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to run on")

    # SAM generator hyperparameters
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.5)
    parser.add_argument("--stability-score-thresh", type=float, default=0.5)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--min-mask-region-area", type=int, default=100)

    # Selection and labeling
    parser.add_argument("--select-id", type=int, default=None, help="Mask ID to save")
    parser.add_argument("--select-largest", action="store_true", help="Automatically pick the largest mask by area")
    parser.add_argument("--label", type=str, default="mobile_phone", help="Annotation label to store in JSON")
    parser.add_argument("--show", action="store_true", help="Show preview window interactively")
    parser.add_argument("--verbose", action="store_true", help="Print verbose progress and diagnostics")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        print("[INFO] Starting SAM annotation utility", flush=True)
    sam, _ = load_sam(args.model_type, args.checkpoint, args.device, args.verbose)
    mask_generator = build_mask_generator(
        sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        min_mask_region_area=args.min_mask_region_area,
    )

    images = list_images(args.input)
    if args.verbose:
        print(f"[INFO] Found {len(images)} image(s) under: {args.input}", flush=True)
    if not images:
        print(f"[ERROR] No images found at: {args.input}")
        return

    selection_strategy = "largest" if args.select_largest else None
    for img_path in images:
        if args.verbose:
            print(f"[INFO] Processing: {img_path}", flush=True)
        process_image(
            image_path=img_path,
            mask_generator=mask_generator,
            out_dir=args.output,
            select_id=args.select_id,
            select_strategy=selection_strategy,
            label=args.label,
            show=args.show,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
