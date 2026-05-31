from pathlib import Path
import csv
import json
import re

import easyocr
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OCR_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "ocr_dataset"
    / "synthetic_plates"
)

IMAGES_DIR = OCR_DATASET_DIR / "images"
MANIFEST_PATH = OCR_DATASET_DIR / "synthetic_plate_manifest.csv"

PUBLIC_METRICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "ocr"
)

METRICS_JSON_PATH = PUBLIC_METRICS_DIR / "synthetic_ocr_metrics.json"
METRICS_CSV_PATH = PUBLIC_METRICS_DIR / "synthetic_ocr_metrics.csv"

VALID_PLATE_PATTERN = re.compile(r"^[0-8][0-9][A-Z]{1,3}[0-9]{2,4}$")


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

    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


def character_accuracy(target: str, prediction: str) -> float:
   
    max_length = max(len(target), len(prediction))

    if max_length == 0:
        return 1.0

    correct = 0

    for index in range(max_length):
        target_char = target[index] if index < len(target) else ""
        prediction_char = prediction[index] if index < len(prediction) else ""

        if target_char == prediction_char:
            correct += 1

    return correct / max_length


def read_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Synthetic OCR manifest not found: {MANIFEST_PATH}"
        )

    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def extract_plate_text(ocr_result) -> str:
    if not ocr_result:
        return ""

    fragments = []

    for item in ocr_result:
        box, text, confidence = item

        x_values = [point[0] for point in box]
        y_values = [point[1] for point in box]

        x_center = sum(x_values) / len(x_values)
        y_center = sum(y_values) / len(y_values)

        normalized_text = normalize_ocr_text(text)
        if x_center < 70 and normalized_text in {"TR", "T", "R"}:
            continue

        if not normalized_text:
            continue

        fragments.append(
            {
                "text": normalized_text,
                "x_center": x_center,
                "y_center": y_center,
                "confidence": confidence,
            }
        )

    fragments.sort(key=lambda item: item["x_center"])

    return "".join(fragment["text"] for fragment in fragments)


def save_metrics(summary: dict) -> None:
    PUBLIC_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with METRICS_CSV_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "samples",
                "exact_match_accuracy",
                "mean_character_accuracy",
                "valid_format_rate",
                "gpu_used",
            ],
        )

        writer.writeheader()
        writer.writerow(
            {
                "samples": summary["samples"],
                "exact_match_accuracy": summary["exact_match_accuracy"],
                "mean_character_accuracy": summary["mean_character_accuracy"],
                "valid_format_rate": summary["valid_format_rate"],
                "gpu_used": summary["gpu_used"],
            }
        )


def main() -> None:
    print(" Synthetic Turkish Plate OCR Evaluation ")

    rows = read_manifest()

    gpu_available = torch.cuda.is_available()

    print(f"Synthetic samples: {len(rows)}")
    print(f"EasyOCR GPU:       {gpu_available}")

    reader = easyocr.Reader(["en"], gpu=gpu_available)

    exact_matches = 0
    character_scores = []
    valid_format_count = 0

    for index, row in enumerate(rows, start=1):
        image_path = IMAGES_DIR / row["file_name"]
        target = row["normalized_plate_text"]

        result = reader.readtext(
            str(image_path),
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            decoder="greedy",
            text_threshold=0.4,
            low_text=0.3,
            width_ths=1.0,
        )

        raw_prediction = extract_plate_text(result)
        prediction = normalize_ocr_text(raw_prediction)

        if prediction == target:
            exact_matches += 1

        character_scores.append(
            character_accuracy(target, prediction)
        )

        if VALID_PLATE_PATTERN.match(prediction):
            valid_format_count += 1

        if index % 50 == 0:
            print(f"Processed {index}/{len(rows)} samples")

    sample_count = len(rows)

    summary = {
        "task": "synthetic_turkish_plate_ocr",
        "ocr_engine": "EasyOCR",
        "language_setting": ["en"],
        "samples": sample_count,
        "exact_match_accuracy": round(exact_matches / sample_count, 6),
        "mean_character_accuracy": round(
            sum(character_scores) / sample_count,
            6,
        ),
        "valid_format_rate": round(
            valid_format_count / sample_count,
            6,
        ),
        "gpu_used": gpu_available,
        "privacy_note": (
            "OCR metrics are computed on synthetically generated plate images. "
            "No real plate identifiers are written to public result files."
        ),
    }

    save_metrics(summary)

    print("\n Privacy-Safe OCR Metrics ")
    print(f"Samples:                  {summary['samples']}")
    print(f"Exact match accuracy:      {summary['exact_match_accuracy']:.4f}")
    print(f"Mean character accuracy:   {summary['mean_character_accuracy']:.4f}")
    print(f"Valid format rate:         {summary['valid_format_rate']:.4f}")

    print("\nSafe OCR metric files saved to:")
    print(METRICS_JSON_PATH)
    print(METRICS_CSV_PATH)


if __name__ == "__main__":
    main()