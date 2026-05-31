from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import re

import cv2
import easyocr
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WEIGHTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "private"
    / "training_runs"
    / "yolo26n_training"
    / "weights"
    / "best.pt"
)

DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "prepared_dataset"
    / "test"
    / "images"
)

PRIVATE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "private"
    / "full_pipeline"
    / "yolo26n_test_samples"
)

PUBLIC_METRICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "pipeline"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

VALID_PLATE_PATTERN = re.compile(r"^[0-8][0-9][A-Z]{1,3}[0-9]{2,4}$")


def parse_arguments():
    parser = ArgumentParser(
        description="Run YOLO detector + OCR pipeline on local license plate images."
    )

    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS_PATH),
        help="Path to trained YOLO detector weights.",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing input images.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum number of images to process.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="YOLO confidence threshold.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size.",
    )

    parser.add_argument(
        "--crop-padding",
        type=int,
        default=4,
        help="Padding in pixels added around detected plate crops.",
    )

    parser.add_argument(
        "--save-private-crops",
        action="store_true",
        help="Save detected plate crops locally under results/private.",
    )

    return parser.parse_args()


def normalize_ocr_text(text: str) -> str:
    text = text.upper()

    replacements = {
        " ": "",
        "-": "",
        ".": "",
        ":": "",
        "_": "",
        "|": "I",
    }

    for old_value, new_value in replacements.items():
        text = text.replace(old_value, new_value)

    return re.sub(r"[^A-Z0-9]", "", text)

def preprocess_crop_variants(crop_image):
    variants = []

    if crop_image is None or crop_image.size == 0:
        return variants

    variants.append(("original", crop_image))

    scale_factor = 3
    enlarged = cv2.resize(
        crop_image,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC,
    )
    variants.append(("enlarged", enlarged))

    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    variants.append(("gray_enlarged", gray))

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    _, otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    variants.append(("otsu_threshold", otsu))

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    variants.append(("adaptive_threshold", adaptive))

    sharpen_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ]
    )
    sharpened = cv2.filter2D(enlarged, -1, sharpen_kernel)
    variants.append(("sharpened", sharpened))

    return variants

def extract_plate_text(ocr_result) -> str:
    if not ocr_result:
        return ""

    fragments = []

    for item in ocr_result:
        box, text, confidence = item

        x_values = [point[0] for point in box]
        x_center = sum(x_values) / len(x_values)

        normalized_text = normalize_ocr_text(text)

        if x_center < 70 and normalized_text in {"TR", "T", "R"}:
            continue

        if normalized_text:
            fragments.append(
                {
                    "text": normalized_text,
                    "x_center": x_center,
                    "confidence": confidence,
                }
            )

    fragments.sort(key=lambda item: item["x_center"])

    return "".join(fragment["text"] for fragment in fragments)


def find_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def crop_with_padding(
    image,
    box_xyxy,
    padding: int,
):
    height, width = image.shape[:2]

    x1, y1, x2, y2 = [int(value) for value in box_xyxy]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)

    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def run_ocr(reader, crop_image) -> tuple[str, str]:
    candidates = []

    for variant_name, variant_image in preprocess_crop_variants(crop_image):
        result = reader.readtext(
            variant_image,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            decoder="greedy",
            text_threshold=0.3,
            low_text=0.2,
            width_ths=1.2,
        )

        raw_text = extract_plate_text(result)
        prediction = normalize_ocr_text(raw_text)

        if not prediction:
            continue

        candidates.append(
            {
                "text": prediction,
                "variant": variant_name,
                "is_valid_format": bool(VALID_PLATE_PATTERN.match(prediction)),
                "length": len(prediction),
            }
        )

    if not candidates:
        return "", "none"

    valid_candidates = [
        candidate
        for candidate in candidates
        if candidate["is_valid_format"]
    ]

    if valid_candidates:
        selected = max(
            valid_candidates,
            key=lambda candidate: candidate["length"],
        )
    else:
        selected = max(
            candidates,
            key=lambda candidate: candidate["length"],
        )

    return selected["text"], selected["variant"]


def save_private_results(rows: list[dict]) -> Path:
    PRIVATE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    private_csv_path = PRIVATE_OUTPUT_DIR / "private_pipeline_predictions.csv"

    with private_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_name",
                "detection_index",
                "detection_confidence",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "ocr_prediction_private",
                "ocr_variant",
                "valid_turkish_plate_format",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return private_csv_path


def save_public_summary(summary: dict) -> tuple[Path, Path]:
    PUBLIC_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = PUBLIC_METRICS_DIR / "full_pipeline_summary.json"
    csv_path = PUBLIC_METRICS_DIR / "full_pipeline_summary.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "images_processed",
                "detections",
                "images_with_detection",
                "mean_detections_per_image",
                "valid_format_rate",
                "gpu_used",
                "detector",
                "ocr_engine",
            ],
        )

        writer.writeheader()
        writer.writerow(
            {
                "images_processed": summary["images_processed"],
                "detections": summary["detections"],
                "images_with_detection": summary["images_with_detection"],
                "mean_detections_per_image": summary["mean_detections_per_image"],
                "valid_format_rate": summary["valid_format_rate"],
                "gpu_used": summary["gpu_used"],
                "detector": summary["detector"],
                "ocr_engine": summary["ocr_engine"],
            }
        )

    return json_path, csv_path


def main() -> None:
    args = parse_arguments()

    weights_path = Path(args.weights).resolve()
    input_dir = Path(args.input_dir).resolve()

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = find_images(input_dir)

    if args.limit > 0:
        images = images[: args.limit]

    gpu_available = torch.cuda.is_available()

    print(" Full License Plate Recognition Pipeline ")
    print(f"Detector weights: {weights_path}")
    print(f"Input directory:  {input_dir}")
    print(f"Images selected:  {len(images)}")
    print(f"EasyOCR GPU:      {gpu_available}")
    print(f"Save crops:       {args.save_private_crops}\n")

    model = YOLO(str(weights_path))
    reader = easyocr.Reader(["en"], gpu=gpu_available)

    private_rows = []

    images_with_detection = 0
    total_detections = 0
    valid_format_count = 0

    crops_dir = PRIVATE_OUTPUT_DIR / "crops"

    if args.save_private_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    for image_index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARNING] Could not read image: {image_path}")
            continue

        prediction = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=0 if gpu_available else "cpu",
            verbose=False,
        )[0]

        boxes = prediction.boxes

        if boxes is None or len(boxes) == 0:
            print(f"[{image_index}/{len(images)}] {image_path.name}: no detection")
            continue

        images_with_detection += 1

        for detection_index, box in enumerate(boxes, start=1):
            confidence = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()

            crop, clipped_box = crop_with_padding(
                image,
                xyxy,
                args.crop_padding,
            )

            if crop.size == 0:
                continue

            ocr_prediction, ocr_variant = run_ocr(reader, crop)
            is_valid_format = bool(VALID_PLATE_PATTERN.match(ocr_prediction))

            if is_valid_format:
                valid_format_count += 1

            total_detections += 1

            x1, y1, x2, y2 = clipped_box

            if args.save_private_crops:
                crop_name = (
                    f"{image_path.stem}_det{detection_index}_private_crop.png"
                )
                cv2.imwrite(str(crops_dir / crop_name), crop)

            private_rows.append(
                {
                    "image_name": image_path.name,
                    "detection_index": detection_index,
                    "detection_confidence": round(confidence, 6),
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "ocr_prediction_private": ocr_prediction,
                    "ocr_variant": ocr_variant,
                    "valid_turkish_plate_format": is_valid_format,
                }
            )

        print(
            f"[{image_index}/{len(images)}] {image_path.name}: "
            f"{len(boxes)} detection(s)"
        )

    private_csv_path = save_private_results(private_rows)

    valid_format_rate = (
        valid_format_count / total_detections
        if total_detections > 0
        else 0.0
    )

    mean_detections_per_image = (
        total_detections / len(images)
        if images
        else 0.0
    )

    public_summary = {
        "task": "full_license_plate_detection_and_ocr_pipeline",
        "detector": "YOLO26n",
        "ocr_engine": "EasyOCR",
        "images_processed": len(images),
        "detections": total_detections,
        "images_with_detection": images_with_detection,
        "mean_detections_per_image": round(mean_detections_per_image, 6),
        "valid_format_rate": round(valid_format_rate, 6),
        "gpu_used": gpu_available,
        "privacy_note": (
            "Readable OCR predictions from real plate crops are saved only "
            "under results/private and are excluded from Git. Public metrics "
            "contain aggregate counts only."
        ),
    }

    json_path, csv_path = save_public_summary(public_summary)

    print("\n=== Private Full Pipeline Output ===")
    print(private_csv_path)

    print("\n=== Public Aggregate Summary ===")
    print(f"Images processed:          {public_summary['images_processed']}")
    print(f"Detections:                {public_summary['detections']}")
    print(f"Images with detection:     {public_summary['images_with_detection']}")
    print(f"Mean detections per image: {public_summary['mean_detections_per_image']:.4f}")
    print(f"Valid format rate:         {public_summary['valid_format_rate']:.4f}")

    print("\nPublic summary files saved to:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()