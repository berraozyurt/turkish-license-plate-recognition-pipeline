from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "detection_dataset"
SUMMARY_PATH = PROJECT_ROOT / "results" / "dataset_checks" / "dataset_summary.json"

SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASS_ID = 0


def find_images(images_dir: Path) -> list[Path]:
    """Return all image files in one split."""
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_labels(labels_dir: Path) -> list[Path]:
    """Return all annotation text files in one split."""
    return sorted(
        path
        for path in labels_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".txt"
    )


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash for exact duplicate detection."""
    hash_object = sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hash_object.update(chunk)

    return hash_object.hexdigest()


def validate_detection_box(values: list[float]) -> bool:
    """
    Validate one YOLO detection box.

    Expected format after the class id:
    x_center y_center width height
    """
    if len(values) != 4:
        return False

    x_center, y_center, width, height = values

    return (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    )


def validate_polygon(points: list[float]) -> bool:
    """
    Validate one normalized polygon annotation.

    Expected format after the class id:
    x1 y1 x2 y2 x3 y3 ...
    """
    if len(points) < 6:
        return False

    if len(points) % 2 != 0:
        return False

    coordinates_are_normalized = all(
        0.0 <= value <= 1.0
        for value in points
    )

    if not coordinates_are_normalized:
        return False

    x_values = points[0::2]
    y_values = points[1::2]

    polygon_width = max(x_values) - min(x_values)
    polygon_height = max(y_values) - min(y_values)

    return polygon_width > 0.0 and polygon_height > 0.0


def inspect_label_file(label_path: Path) -> dict[str, int]:
    """
    Inspect one annotation file.

    The downloaded dataset contains a mixture of:
    - YOLO bounding-box lines: class_id x_center y_center width height
    - YOLO polygon lines: class_id x1 y1 x2 y2 ...
    """
    file_summary = {
        "objects": 0,
        "bounding_box_annotations": 0,
        "polygon_annotations": 0,
        "invalid_lines": 0,
        "empty_files": 0,
    }

    nonempty_lines = 0

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()

            if not stripped_line:
                continue

            nonempty_lines += 1
            values = stripped_line.split()

            try:
                class_value = float(values[0])
                annotation_values = [
                    float(value)
                    for value in values[1:]
                ]
            except (ValueError, IndexError):
                file_summary["invalid_lines"] += 1
                continue

            if not class_value.is_integer() or int(class_value) != EXPECTED_CLASS_ID:
                file_summary["invalid_lines"] += 1
                continue

            if len(values) == 5 and validate_detection_box(annotation_values):
                file_summary["objects"] += 1
                file_summary["bounding_box_annotations"] += 1
                continue

            if len(values) >= 7 and validate_polygon(annotation_values):
                file_summary["objects"] += 1
                file_summary["polygon_annotations"] += 1
                continue

            file_summary["invalid_lines"] += 1

    if nonempty_lines == 0:
        file_summary["empty_files"] = 1

    return file_summary


def inspect_split(split_name: str) -> tuple[dict, dict[str, str]]:
    """Inspect image-label consistency and annotation types in one split."""
    split_dir = DATASET_DIR / split_name
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    images = find_images(images_dir)
    labels = find_labels(labels_dir)

    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    images_without_labels = image_stems - label_stems
    labels_without_images = label_stems - image_stems

    total_objects = 0
    bounding_box_annotations = 0
    polygon_annotations = 0
    invalid_lines = 0
    empty_label_files = 0

    for label_path in labels:
        label_summary = inspect_label_file(label_path)

        total_objects += label_summary["objects"]
        bounding_box_annotations += label_summary["bounding_box_annotations"]
        polygon_annotations += label_summary["polygon_annotations"]
        invalid_lines += label_summary["invalid_lines"]
        empty_label_files += label_summary["empty_files"]

    image_hashes = {
        calculate_file_hash(image_path): split_name
        for image_path in images
    }

    split_summary = {
        "images": len(images),
        "labels": len(labels),
        "objects": total_objects,
        "bounding_box_annotations": bounding_box_annotations,
        "polygon_annotations": polygon_annotations,
        "empty_label_files": empty_label_files,
        "images_without_labels": len(images_without_labels),
        "labels_without_images": len(labels_without_images),
        "invalid_lines": invalid_lines,
    }

    return split_summary, image_hashes


def find_cross_split_duplicates(
    hashes_by_split: dict[str, dict[str, str]],
) -> int:
    """Count exact image files occurring in more than one split."""
    split_occurrences = defaultdict(set)

    for split_name, hash_map in hashes_by_split.items():
        for file_hash in hash_map:
            split_occurrences[file_hash].add(split_name)

    return sum(
        1
        for split_names in split_occurrences.values()
        if len(split_names) > 1
    )


def main() -> None:
    print("=== Detection Dataset Validation ===")
    print(f"Dataset directory: {DATASET_DIR}\n")

    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    yaml_path = DATASET_DIR / "data.yaml"

    if not yaml_path.exists():
        print(f"[ERROR] data.yaml not found: {yaml_path}")
        sys.exit(1)

    split_summaries = {}
    hashes_by_split = {}

    try:
        for split_name in SPLITS:
            split_summary, image_hashes = inspect_split(split_name)
            split_summaries[split_name] = split_summary
            hashes_by_split[split_name] = image_hashes

    except FileNotFoundError as error:
        print(f"[ERROR] {error}")
        sys.exit(1)

    cross_split_duplicates = find_cross_split_duplicates(hashes_by_split)

    total_images = sum(row["images"] for row in split_summaries.values())
    total_labels = sum(row["labels"] for row in split_summaries.values())
    total_objects = sum(row["objects"] for row in split_summaries.values())
    total_boxes = sum(
        row["bounding_box_annotations"]
        for row in split_summaries.values()
    )
    total_polygons = sum(
        row["polygon_annotations"]
        for row in split_summaries.values()
    )
    total_empty_labels = sum(
        row["empty_label_files"]
        for row in split_summaries.values()
    )
    total_missing_labels = sum(
        row["images_without_labels"]
        for row in split_summaries.values()
    )
    total_orphan_labels = sum(
        row["labels_without_images"]
        for row in split_summaries.values()
    )
    total_invalid_lines = sum(
        row["invalid_lines"]
        for row in split_summaries.values()
    )

    print(
        "Split   | Images | Labels | Objects | Boxes | Polygons | "
        "Empty Labels | Invalid Lines"
    )
    print("-" * 92)

    for split_name in SPLITS:
        row = split_summaries[split_name]

        print(
            f"{split_name:<7} | "
            f"{row['images']:>6} | "
            f"{row['labels']:>6} | "
            f"{row['objects']:>7} | "
            f"{row['bounding_box_annotations']:>5} | "
            f"{row['polygon_annotations']:>8} | "
            f"{row['empty_label_files']:>12} | "
            f"{row['invalid_lines']:>13}"
        )

    print("\n=== Overall Dataset Summary ===")
    print(f"Total images:                    {total_images}")
    print(f"Total label files:               {total_labels}")
    print(f"Total annotated objects:         {total_objects}")
    print(f"Bounding-box annotations:        {total_boxes}")
    print(f"Polygon annotations:             {total_polygons}")
    print(f"Empty label files:               {total_empty_labels}")
    print(f"Images without labels:           {total_missing_labels}")
    print(f"Labels without images:           {total_orphan_labels}")
    print(f"Invalid annotation lines:        {total_invalid_lines}")
    print(f"Exact duplicates across splits:  {cross_split_duplicates}")

    needs_review = (
        total_missing_labels > 0
        or total_orphan_labels > 0
        or total_invalid_lines > 0
        or cross_split_duplicates > 0
    )

    summary = {
        "dataset": "License Plates of Vehicles in Turkey",
        "task_used_in_project": "object_detection",
        "class_names": ["license plate"],
        "source_annotation_types": [
            "bounding_box",
            "polygon",
        ],
        "conversion_plan": (
            "Polygon annotations will be converted to bounding boxes "
            "for YOLO object detection training."
        ),
        "source_split_summary": split_summaries,
        "overall": {
            "images": total_images,
            "label_files": total_labels,
            "annotated_objects": total_objects,
            "bounding_box_annotations": total_boxes,
            "polygon_annotations": total_polygons,
            "empty_label_files": total_empty_labels,
            "images_without_labels": total_missing_labels,
            "labels_without_images": total_orphan_labels,
            "invalid_annotation_lines": total_invalid_lines,
            "exact_duplicates_across_splits": cross_split_duplicates,
        },
        "status": "needs_review" if needs_review else "valid_for_conversion",
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"\nPrivacy-safe summary saved to: {SUMMARY_PATH}")

    if needs_review:
        print("\n[WARNING] Dataset needs review before conversion.")
    else:
        print(
            "\nDataset is valid for conversion to YOLO detection "
            "bounding-box format."
        )


if __name__ == "__main__":
    main()