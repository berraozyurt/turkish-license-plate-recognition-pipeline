from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import sys

import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_YAML_PATH = (
    PROJECT_ROOT
    / "data"
    / "prepared_dataset"
    / "data.yaml"
)

PUBLIC_METRICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
    / "detection"
)

PRIVATE_EVALUATION_DIR = (
    PROJECT_ROOT
    / "results"
    / "private"
    / "evaluation_runs"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLIT_FOLDER_NAMES = {
    "val": "valid",
    "test": "test",
}


def parse_arguments():
    parser = ArgumentParser(
        description="Evaluate a trained YOLO license plate detector."
    )

    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment identifier, for example yolo26n or yolo11n.",
    )

    parser.add_argument(
        "--weights",
        required=True,
        type=str,
        help="Path to the trained best.pt checkpoint.",
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Dataset split to evaluate. Validation is used during model comparison.",
    )

    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Required before evaluating the held-out test set.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Evaluation image size.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Data loading workers. Zero is safe on Windows.",
    )

    return parser.parse_args()


def resolve_weights_path(weights_argument: str) -> Path:
    weights_path = Path(weights_argument)

    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path

    return weights_path.resolve()


def validate_inputs(arguments, weights_path: Path) -> None:
    if not DATA_YAML_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset configuration not found: {DATA_YAML_PATH}"
        )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}"
        )

    if arguments.split == "test" and not arguments.confirm_final_test:
        raise RuntimeError(
            "Test evaluation is protected. "
            "Use --confirm-final-test only after selecting the final detector."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was not detected. Evaluation is configured for GPU use."
        )


def count_split_items(split_name: str) -> tuple[int, int]:
    """Count images and annotated objects without exposing image contents."""
    dataset_split_name = SPLIT_FOLDER_NAMES[split_name]

    images_dir = (
        PROJECT_ROOT
        / "data"
        / "prepared_dataset"
        / dataset_split_name
        / "images"
    )

    labels_dir = (
        PROJECT_ROOT
        / "data"
        / "prepared_dataset"
        / dataset_split_name
        / "labels"
    )

    image_count = sum(
        1
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    object_count = 0

    for label_path in labels_dir.glob("*.txt"):
        with label_path.open("r", encoding="utf-8") as file:
            object_count += sum(
                1
                for line in file
                if line.strip()
            )

    return image_count, object_count


def extract_metric(metric_dictionary: dict, metric_name: str) -> float:
    """Read one metric value from the Ultralytics result dictionary."""
    metric_name_lower = metric_name.lower()

    for key, value in metric_dictionary.items():
        key_lower = key.lower()

        if metric_name_lower == "map50":
            if "map50" in key_lower and "map50-95" not in key_lower:
                return float(value)

        elif metric_name_lower in key_lower:
            return float(value)

    raise KeyError(
        f"Could not find metric '{metric_name}' in: {metric_dictionary}"
    )


def save_safe_metrics(record: dict) -> tuple[Path, Path]:
    """Save aggregate metrics only; no image or readable plate data."""
    PUBLIC_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = (
        PUBLIC_METRICS_DIR
        / f"{record['experiment']}_{record['split']}_metrics.json"
    )

    csv_path = PUBLIC_METRICS_DIR / "model_comparison_metrics.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)

    fieldnames = [
        "experiment",
        "split",
        "images",
        "objects",
        "precision",
        "recall",
        "map50",
        "map50_95",
        "preprocess_ms_per_image",
        "inference_ms_per_image",
        "postprocess_ms_per_image",
        "imgsz",
        "batch",
    ]

    previous_rows = []

    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as file:
            previous_rows = list(csv.DictReader(file))

    previous_rows = [
        row
        for row in previous_rows
        if not (
            row["experiment"] == record["experiment"]
            and row["split"] == record["split"]
        )
    ]

    previous_rows.append(
        {
            field_name: record[field_name]
            for field_name in fieldnames
        }
    )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(previous_rows)

    return json_path, csv_path


def main() -> None:
    arguments = parse_arguments()
    weights_path = resolve_weights_path(arguments.weights)

    try:
        validate_inputs(arguments, weights_path)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"[ERROR] {error}")
        sys.exit(1)

    image_count, object_count = count_split_items(arguments.split)

    PRIVATE_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    print("=== YOLO Detector Evaluation ===")
    print(f"Experiment:     {arguments.experiment}")
    print(f"Weights:        {weights_path}")
    print(f"Dataset config: {DATA_YAML_PATH}")
    print(f"Split:          {arguments.split}")
    print(f"Images:         {image_count}")
    print(f"Objects:        {object_count}")
    print(f"GPU:            {torch.cuda.get_device_name(0)}\n")

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(DATA_YAML_PATH),
        split=arguments.split,
        imgsz=arguments.imgsz,
        batch=arguments.batch,
        device=0,
        workers=arguments.workers,
        project=str(PRIVATE_EVALUATION_DIR),
        name=f"{arguments.experiment}_{arguments.split}",
        exist_ok=True,
        plots=False,
        save_json=False,
        verbose=True,
    )

    metric_dictionary = metrics.results_dict

    record = {
        "experiment": arguments.experiment,
        "split": arguments.split,
        "images": image_count,
        "objects": object_count,
        "precision": round(
            extract_metric(metric_dictionary, "precision"),
            6,
        ),
        "recall": round(
            extract_metric(metric_dictionary, "recall"),
            6,
        ),
        "map50": round(
            extract_metric(metric_dictionary, "map50"),
            6,
        ),
        "map50_95": round(
            extract_metric(metric_dictionary, "map50-95"),
            6,
        ),
        "preprocess_ms_per_image": round(
            float(metrics.speed.get("preprocess", 0.0)),
            4,
        ),
        "inference_ms_per_image": round(
            float(metrics.speed.get("inference", 0.0)),
            4,
        ),
        "postprocess_ms_per_image": round(
            float(metrics.speed.get("postprocess", 0.0)),
            4,
        ),
        "imgsz": arguments.imgsz,
        "batch": arguments.batch,
    }

    json_path, csv_path = save_safe_metrics(record)

    print("\n=== Privacy-Safe Aggregate Metrics ===")
    print(f"Precision:  {record['precision']:.4f}")
    print(f"Recall:     {record['recall']:.4f}")
    print(f"mAP50:      {record['map50']:.4f}")
    print(f"mAP50-95:   {record['map50_95']:.4f}")
    print(f"Inference:  {record['inference_ms_per_image']:.4f} ms/image")

    print("\nSafe metric files saved to:")
    print(json_path)
    print(csv_path)

    print(
        "\nNo raw prediction images or readable plate identifiers "
        "were written to the public metrics directory."
    )


if __name__ == "__main__":
    main()