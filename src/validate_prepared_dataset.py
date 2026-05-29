from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import csv
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "prepared_dataset"
MANIFEST_PATH = DATASET_DIR / "split_manifest.csv"

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "dataset_checks"
    / "prepared_dataset_validation.json"
)

SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASS_ID = 0


def find_images(images_dir: Path) -> list[Path]:
    """Return all supported image files in one split."""
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_labels(labels_dir: Path) -> list[Path]:
    """Return all YOLO label files in one split."""
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


def validate_label_file(label_path: Path) -> dict[str, int]:
    """
    Validate one prepared YOLO detection label file.

    Required non-empty line format:
    class_id x_center y_center width height
    """
    statistics = {
        "objects": 0,
        "invalid_lines": 0,
        "empty_label_files": 0,
    }

    nonempty_lines = 0

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()

            if not stripped_line:
                continue

            nonempty_lines += 1
            values = stripped_line.split()

            if len(values) != 5:
                statistics["invalid_lines"] += 1
                continue

            try:
                class_value = float(values[0])
                x_center, y_center, width, height = [
                    float(value)
                    for value in values[1:]
                ]
            except ValueError:
                statistics["invalid_lines"] += 1
                continue

            valid_class = (
                class_value.is_integer()
                and int(class_value) == EXPECTED_CLASS_ID
            )

            valid_coordinates = (
                0.0 <= x_center <= 1.0
                and 0.0 <= y_center <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
            )

            if not valid_class or not valid_coordinates:
                statistics["invalid_lines"] += 1
                continue

            statistics["objects"] += 1

    if nonempty_lines == 0:
        statistics["empty_label_files"] = 1

    return statistics


def inspect_split(split_name: str) -> tuple[dict, dict[str, str]]:
    """Inspect one prepared train, validation or test split."""
    images_dir = DATASET_DIR / split_name / "images"
    labels_dir = DATASET_DIR / split_name / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    images = find_images(images_dir)
    labels = find_labels(labels_dir)

    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    object_count = 0
    invalid_line_count = 0
    empty_label_file_count = 0

    for label_path in labels:
        statistics = validate_label_file(label_path)

        object_count += statistics["objects"]
        invalid_line_count += statistics["invalid_lines"]
        empty_label_file_count += statistics["empty_label_files"]

    image_hashes = {
        calculate_file_hash(image_path): split_name
        for image_path in images
    }

    split_summary = {
        "images": len(images),
        "labels": len(labels),
        "objects": object_count,
        "empty_label_files": empty_label_file_count,
        "images_without_labels": len(missing_labels),
        "labels_without_images": len(orphan_labels),
        "invalid_label_lines": invalid_line_count,
    }

    return split_summary, image_hashes


def count_exact_cross_split_duplicates(
    hashes_by_split: dict[str, dict[str, str]],
) -> int:
    """Count exact image files that occur in more than one split."""
    occurrences = defaultdict(set)

    for split_name, hash_map in hashes_by_split.items():
        for image_hash in hash_map:
            occurrences[image_hash].add(split_name)

    return sum(
        1
        for split_names in occurrences.values()
        if len(split_names) > 1
    )


def inspect_manifest() -> dict[str, int]:
    """Check that one source group is never assigned to multiple output splits."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Split manifest not found: {MANIFEST_PATH}")

    group_to_splits = defaultdict(set)
    rows = 0

    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows += 1
            group_to_splits[row["source_group"]].add(row["output_split"])

    overlapping_groups = sum(
        1
        for split_names in group_to_splits.values()
        if len(split_names) > 1
    )

    return {
        "manifest_rows": rows,
        "source_groups": len(group_to_splits),
        "groups_across_multiple_splits": overlapping_groups,
    }


def main() -> None:
    print("=== Prepared Detection Dataset Validation ===")
    print(f"Prepared dataset: {DATASET_DIR}\n")

    if not DATASET_DIR.exists():
        print(f"[ERROR] Prepared dataset not found: {DATASET_DIR}")
        sys.exit(1)

    yaml_path = DATASET_DIR / "data.yaml"

    if not yaml_path.exists():
        print(f"[ERROR] data.yaml not found: {yaml_path}")
        sys.exit(1)

    try:
        split_summaries = {}
        hashes_by_split = {}

        for split_name in SPLITS:
            split_summary, image_hashes = inspect_split(split_name)
            split_summaries[split_name] = split_summary
            hashes_by_split[split_name] = image_hashes

        manifest_summary = inspect_manifest()

    except FileNotFoundError as error:
        print(f"[ERROR] {error}")
        sys.exit(1)

    exact_duplicates = count_exact_cross_split_duplicates(hashes_by_split)

    overall = {
        "images": sum(row["images"] for row in split_summaries.values()),
        "labels": sum(row["labels"] for row in split_summaries.values()),
        "objects": sum(row["objects"] for row in split_summaries.values()),
        "empty_label_files": sum(
            row["empty_label_files"]
            for row in split_summaries.values()
        ),
        "images_without_labels": sum(
            row["images_without_labels"]
            for row in split_summaries.values()
        ),
        "labels_without_images": sum(
            row["labels_without_images"]
            for row in split_summaries.values()
        ),
        "invalid_label_lines": sum(
            row["invalid_label_lines"]
            for row in split_summaries.values()
        ),
        "exact_duplicates_across_splits": exact_duplicates,
        "source_groups_across_multiple_splits": (
            manifest_summary["groups_across_multiple_splits"]
        ),
    }

    print(
        "Split   | Images | Labels | Objects | Empty Labels | "
        "Missing Labels | Orphan Labels | Invalid Lines"
    )
    print("-" * 104)

    for split_name in SPLITS:
        row = split_summaries[split_name]

        print(
            f"{split_name:<7} | "
            f"{row['images']:>6} | "
            f"{row['labels']:>6} | "
            f"{row['objects']:>7} | "
            f"{row['empty_label_files']:>12} | "
            f"{row['images_without_labels']:>14} | "
            f"{row['labels_without_images']:>13} | "
            f"{row['invalid_label_lines']:>13}"
        )

    print("\n=== Validation Summary ===")
    print(f"Total images:                         {overall['images']}")
    print(f"Total label files:                    {overall['labels']}")
    print(f"Total detection boxes:                {overall['objects']}")
    print(f"Empty label files retained:           {overall['empty_label_files']}")
    print(f"Images without labels:                {overall['images_without_labels']}")
    print(f"Labels without images:                {overall['labels_without_images']}")
    print(f"Invalid YOLO bounding-box lines:      {overall['invalid_label_lines']}")
    print(f"Exact duplicates across splits:       {overall['exact_duplicates_across_splits']}")
    print(f"Source groups across multiple splits: {overall['source_groups_across_multiple_splits']}")
    print(f"Manifest rows:                        {manifest_summary['manifest_rows']}")
    print(f"Unique source groups:                 {manifest_summary['source_groups']}")

    is_valid = (
        overall["images"] == 3500
        and overall["labels"] == 3500
        and overall["objects"] == 6320
        and overall["images_without_labels"] == 0
        and overall["labels_without_images"] == 0
        and overall["invalid_label_lines"] == 0
        and overall["exact_duplicates_across_splits"] == 0
        and overall["source_groups_across_multiple_splits"] == 0
        and manifest_summary["manifest_rows"] == overall["images"]
    )

    safe_summary = {
        "prepared_dataset": "Turkish license plate detection dataset",
        "label_format": "YOLO bounding_box_detection",
        "split_summaries": split_summaries,
        "manifest_summary": manifest_summary,
        "overall": overall,
        "status": "valid_for_training" if is_valid else "needs_review",
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(safe_summary, file, indent=2)

    print(f"\nPrivacy-safe validation summary saved to: {SUMMARY_PATH}")

    if is_valid:
        print("\nPrepared dataset is valid for YOLO detector training.")
    else:
        print("\n[WARNING] Prepared dataset needs review before model training.")
        sys.exit(1)


if __name__ == "__main__":
    main()