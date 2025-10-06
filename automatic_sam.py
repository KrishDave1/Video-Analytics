from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import urllib.request
import torch
import cv2
import numpy as np


SAM_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}


def download_checkpoint_if_needed(model_type: str, checkpoint_path: str, auto_download: bool, verbose: bool) -> str:
    p = Path(checkpoint_path)
    if p.exists():
        return str(p)
    if not auto_download:
        return str(p)
    url = SAM_URLS.get(model_type)
    if url is None:
        return str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[INFO] Downloading {model_type} checkpoint to: {p}", flush=True)
    try:
        urllib.request.urlretrieve(url, str(p))
        if verbose:
            print("[OK] Checkpoint downloaded", flush=True)
    except Exception as e:
        print(f"[WARN] Auto-download failed: {e}", flush=True)
    return str(p)


def load_sam(model_type: str, checkpoint_path: str, device: str, verbose: bool):
    start = time.time()
    if verbose:
        print(f"[INFO] Loading SAM model '{model_type}' from: {checkpoint_path}", flush=True)
    try:
        requested = device
        auto_device = "cuda" if torch.cuda.is_available() else "cpu"
        device_to_use = torch.device(requested if requested in {"cuda", "cpu"} else auto_device)
        if verbose:
            print(f"[INFO] Using device: {device_to_use}", flush=True)
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        try:
            sam.to(device_to_use)
        except AssertionError as e:
            # Common when Torch is CPU-only but user requested cuda
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                print("[WARN] CUDA unavailable; falling back to CPU.", flush=True)
                device_to_use = torch.device("cpu")
                sam.to(device_to_use)
            else:
                raise
        if verbose:
            print(f"[OK] SAM loaded in {time.time()-start:.2f}s", flush=True)
        return sam, device_to_use
    except Exception as e:
        print(f"[ERROR] Failed to load SAM: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


def resolve_preset(preset: str | None) -> dict:
    if preset is None:
        return {}
    presets = {
        "fast": {
            "points_per_side": 16,
            "pred_iou_thresh": 0.5,
            "stability_score_thresh": 0.5,
            "crop_n_layers": 0,
            "min_mask_region_area": 400,
        },
        "balanced": {
            "points_per_side": 32,
            "pred_iou_thresh": 0.5,
            "stability_score_thresh": 0.5,
            "crop_n_layers": 1,
            "min_mask_region_area": 100,
        },
        "quality": {
            "points_per_side": 64,
            "pred_iou_thresh": 0.5,
            "stability_score_thresh": 0.5,
            "crop_n_layers": 2,
            "min_mask_region_area": 50,
        },
    }
    return presets.get(preset, {})


def build_mask_generator(sam, params: dict):
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=int(params["points_per_side"]),
        pred_iou_thresh=float(params["pred_iou_thresh"]),
        stability_score_thresh=float(params["stability_score_thresh"]),
        crop_n_layers=int(params["crop_n_layers"]),
        min_mask_region_area=int(params["min_mask_region_area"]),
    )


def draw_preview(image_bgr: np.ndarray, bboxes: list) -> np.ndarray:
    preview = image_bgr.copy()
    for idx, bbox in enumerate(bboxes):
        x, y, w, h = bbox
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


def resize_to_max_long_edge(image_rgb: np.ndarray, max_long_edge: int) -> tuple[np.ndarray, float]:
    h, w = image_rgb.shape[:2]
    long_edge = max(h, w)
    if max_long_edge <= 0 or long_edge <= max_long_edge:
        return image_rgb, 1.0
    scale = max_long_edge / float(long_edge)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def scale_bbox_to_original(bbox, scale: float) -> list[int]:
    if scale == 1.0:
        return [int(v) for v in bbox]
    x, y, w, h = bbox
    inv = 1.0 / scale
    return [int(round(x * inv)), int(round(y * inv)), int(round(w * inv)), int(round(h * inv))]


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


def interactive_choose_mask(masks: list, top_k: int) -> int | None:
    if not masks:
        print("[WARN] No masks to choose from.")
        return None
    # Prepare sorted indices by area desc
    areas = [(i, m.get("area", 0)) for i, m in enumerate(masks)]
    areas.sort(key=lambda x: x[1], reverse=True)
    show = areas[:max(1, top_k)]
    print("[INFO] Top candidates by area:")
    for rank, (idx, area) in enumerate(show, start=1):
        bbox = masks[idx].get("bbox", [0, 0, 0, 0])
        iou = float(masks[idx].get("predicted_iou", 0.0))
        stab = float(masks[idx].get("stability_score", 0.0))
        print(f"  #{rank:2d}  id={idx:4d}  area={int(area):7d}  bbox={bbox}  iou={iou:.3f}  stab={stab:.3f}")
    try:
        raw = input("Enter mask id to save (blank to skip): ").strip()
    except EOFError:
        return None
    if raw == "":
        return None
    try:
        val = int(raw)
    except ValueError:
        print("[WARN] Invalid id; skipping.")
        return None
    if 0 <= val < len(masks):
        return val
    print("[WARN] Id out of range; skipping.")
    return None


def process_image(image_path: str, mask_generator: SamAutomaticMaskGenerator, out_dir: str,
                  select_id: int | None, select_strategy: str | None, label: str | None, show: bool, verbose: bool,
                  max_long_edge: int, interactive: bool, top_k: int):
    os.makedirs(out_dir, exist_ok=True)
    if verbose:
        print(f"[INFO] Reading image: {image_path}", flush=True)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"[WARN] Could not read image: {image_path}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb, scale = resize_to_max_long_edge(image_rgb, max_long_edge)
    if verbose and scale != 1.0:
        h0, w0 = image_rgb.shape[:2]
        h1, w1 = resized_rgb.shape[:2]
        print(f"[INFO] Resized from {w0}x{h0} to {w1}x{h1} (scale={scale:.3f})", flush=True)

    if verbose:
        print(f"[INFO] Generating masks...", flush=True)
    gen_start = time.time()
    try:
        masks = mask_generator.generate(resized_rgb)
    except Exception as e:
        print(f"[ERROR] Mask generation failed: {e}", flush=True)
        traceback.print_exc()
        return
    if verbose:
        print(f"[OK] Generated {len(masks)} masks in {time.time()-gen_start:.2f}s", flush=True)

    # Save preview with all IDs (scale bboxes back to original size if resized)
    scaled_bboxes = [scale_bbox_to_original(m.get("bbox", [0, 0, 0, 0]), scale) for m in masks]
    preview = draw_preview(image_bgr, scaled_bboxes)
    base = Path(image_path).stem
    per_image_dir = os.path.join(out_dir, base)
    os.makedirs(per_image_dir, exist_ok=True)
    preview_path = os.path.join(per_image_dir, f"{base}_preview.jpg")
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
        if interactive:
            print(f"[INFO] Preview saved at: {preview_path}")
            chosen_id = interactive_choose_mask(masks, top_k)
        if chosen_id is None:
            print(f"[INFO] No selection made. Use --select-id, --select-largest, or --interactive.")
            return

    chosen = masks[chosen_id]
    mask_resized = chosen.get("segmentation").astype(bool)

    # Upscale mask to original size if needed (nearest neighbor to preserve binary)
    if scale != 1.0:
        mask_uint8_small = (mask_resized.astype(np.uint8)) * 255
        h0, w0 = image_bgr.shape[:2]
        mask_uint8_orig = cv2.resize(mask_uint8_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_uint8_orig.astype(bool)
    else:
        mask_bool = mask_resized

    # Save mask PNG and cropped object
    mask_path = os.path.join(per_image_dir, f"{base}_mask_{chosen_id}.png")
    crop_path = os.path.join(per_image_dir, f"{base}_crop_{chosen_id}.png")
    save_mask_png(mask_bool, mask_path)
    save_cropped_object(image_bgr, mask_bool, crop_path)
    if verbose:
        print(f"[OK] Saved mask: {mask_path}", flush=True)
        print(f"[OK] Saved crop: {crop_path}", flush=True)

    # Enrich with polygons and bbox computed at original scale for JSON
    polygons = mask_to_polygons(mask_bool)
    chosen["polygons"] = polygons
    # Compute bbox from mask at original size for accuracy
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        bbox_orig = [int(x), int(y), int(w), int(h)]
    else:
        bbox_orig = scale_bbox_to_original(chosen.get("bbox", [0, 0, 0, 0]), scale)
    chosen["bbox"] = bbox_orig

    # Write JSON annotation
    json_path = os.path.join(per_image_dir, f"{base}_ann_{chosen_id}.json")
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
    parser.add_argument("--preset", choices=["fast", "balanced", "quality"], default="fast", help="Speed/quality preset")
    parser.add_argument("--points-per-side", type=int, default=None)
    parser.add_argument("--pred-iou-thresh", type=float, default=None)
    parser.add_argument("--stability-score-thresh", type=float, default=None)
    parser.add_argument("--crop-n-layers", type=int, default=None)
    parser.add_argument("--min-mask-region-area", type=int, default=None)

    # Selection and labeling
    parser.add_argument("--select-id", type=int, default=None, help="Mask ID to save")
    parser.add_argument("--select-largest", action="store_true", help="Automatically pick the largest mask by area")
    parser.add_argument("--label", type=str, default="mobile_phone", help="Annotation label to store in JSON")
    parser.add_argument("--show", action="store_true", help="Show preview window interactively")
    parser.add_argument("--max-long-edge", type=int, default=1024, help="Resize so max(h,w) <= this value (0 disables)")
    parser.add_argument("--auto-download", action="store_true", help="Auto-download checkpoint if missing")
    parser.add_argument("--interactive", action="store_true", help="Prompt to choose mask id in the terminal")
    parser.add_argument("--top-k", type=int, default=20, help="How many top masks by area to list for selection")
    parser.add_argument("--verbose", action="store_true", help="Print verbose progress and diagnostics")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        print("[INFO] Starting SAM annotation utility", flush=True)
    ckpt_path = download_checkpoint_if_needed(args.model_type, args.checkpoint, args.auto_download, args.verbose)
    sam, _ = load_sam(args.model_type, ckpt_path, args.device, args.verbose)
    # Resolve parameters from preset, allow explicit flags to override when not None
    params = resolve_preset(args.preset)
    if args.points_per_side is not None:
        params["points_per_side"] = args.points_per_side
    if args.pred_iou_thresh is not None:
        params["pred_iou_thresh"] = args.pred_iou_thresh
    if args.stability_score_thresh is not None:
        params["stability_score_thresh"] = args.stability_score_thresh
    if args.crop_n_layers is not None:
        params["crop_n_layers"] = args.crop_n_layers
    if args.min_mask_region_area is not None:
        params["min_mask_region_area"] = args.min_mask_region_area
    if args.verbose:
        print(f"[INFO] SAM params: {params}", flush=True)
    mask_generator = build_mask_generator(sam, params)

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
            max_long_edge=args.max_long_edge,
            interactive=args.interactive,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
