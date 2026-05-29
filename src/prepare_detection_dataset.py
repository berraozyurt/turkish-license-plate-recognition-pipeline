from collections import defaultdict
from pathlib import Path
import csv
import json
import random
import shutil
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DATASET_DIR = PROJECT_ROOT / "data" / "detection_dataset"
OUTPUT_DATASET_DIR = PROJECT_ROOT / "data" / "prepared_dataset"

SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "dataset_checks"
    / "prepared_dataset_summary.json"
)

SOURCE_SPLITS = ["train", "valid", "test"]
OUTPUT_SPLITS = ["train", "valid", "test"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

CLASS_ID = 0
CLASS_NAME = "license plate"


def find_image_for_label(
    images_dir: Path,
    label_stem: str,
) -> Path | None:
    """Find the image file matching one annotation file stem."""
    for extension in IMAGE_EXTENSIONS:
        image_path = images_dir / f"{label_stem}{extension}"

        if image_path.exists():
            return image_path

    return None


def get_source_group_key(image_path: Path) -> str:
    """
    Derive a grouping key from a Roboflow-exported image name.

    Files derived from the same source image usually share the part
    before '.rf.'. Keeping them in the same split reduces leakage risk.
    """
    return image_path.stem.split(".rf.")[0]


def read_source_records() -> list[dict]:
    """Collect source image-label pairs from all downloaded splits."""
    records = []

    for source_split in SOURCE_SPLITS:
        images_dir = SOURCE_DATASET_DIR / source_split / "images"
        labels_dir = SOURCE_DATASET_DIR / source_split / "labels"

        if not images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {images_dir}"
            )

        if not labels_dir.exists():
            raise FileNotFoundError(
                f"Labels directory not found: {labels_dir}"
            )

        label_paths = sorted(labels_dir.glob("*.txt"))

        for label_path in label_paths:
            image_path = find_image_for_label(
                images_dir,
                label_path.stem,
            )

            if image_path is None:
                raise FileNotFoundError(
                    f"No image found for annotation: {label_path}"
                )

            records.append(
                {
                    "image_path": image_path,
                    "label_path": label_path,
                    "source_split": source_split,
                    "group_key": get_source_group_key(image_path),
                }
            )

    return records


def convert_annotation_line(line: str) -> tuple[str | None, str]:
    """
    Convert one annotation line into YOLO detection bounding-box format.

    Returns:
        converted_line: output line or None for blank input
        annotation_type: 'box', 'polygon', or 'empty'
    """
    stripped_line = line.strip()

    if not stripped_line:
        return None, "empty"

    values = stripped_line.split()

    try:
        class_value = int(float(values[0]))
        coordinates = [float(value) for value in values[1:]]
    except (ValueError, IndexError) as error:
        raise ValueError(
            f"Invalid annotation line: {stripped_line}"
        ) from error

    if class_value != CLASS_ID:
        raise ValueError(
            f"Unexpected class id {class_value}: {stripped_line}"
        )

    # Existing YOLO bounding-box annotation:
    # class_id x_center y_center width height
    if len(coordinates) == 4:
        x_center, y_center, width, height = coordinates

        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise ValueError(
                f"Invalid bounding-box annotation: {stripped_line}"
            )

        converted_line = (
            f"{CLASS_ID} "
            f"{x_center:.8f} {y_center:.8f} "
            f"{width:.8f} {height:.8f}"
        )

        return converted_line, "box"

    # Polygon annotation:
    # class_id x1 y1 x2 y2 ... xn yn
    if len(coordinates) < 6 or len(coordinates) % 2 != 0:
        raise ValueError(
            f"Invalid polygon annotation: {stripped_line}"
        )

    if not all(0.0 <= value <= 1.0 for value in coordinates):
        raise ValueError(
            f"Polygon coordinates outside normalized range: {stripped_line}"
        )

    x_values = coordinates[0::2]
    y_values = coordinates[1::2]

    x_min = min(x_values)
    y_min = min(y_values)
    x_max = max(x_values)
    y_max = max(y_values)

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0.0 or height <= 0.0:
        raise ValueError(
            f"Polygon has invalid area: {stripped_line}"
        )

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    converted_line = (
        f"{CLASS_ID} "
        f"{x_center:.8f} {y_center:.8f} "
        f"{width:.8f} {height:.8f}"
    )

    return converted_line, "polygon"


def convert_label_file(
    source_label_path: Path,
    output_label_path: Path,
) -> dict[str, int]:
    """Convert one source annotation file to detection box format."""
    statistics = {
        "objects": 0,
        "existing_boxes": 0,
        "converted_polygons": 0,
        "empty_label_files": 0,
    }

    output_lines = []

    with source_label_path.open("r", encoding="utf-8") as source_file:
        for line in source_file:
            converted_line, annotation_type = convert_annotation_line(line)

            if converted_line is None:
                continue

            output_lines.append(converted_line)
            statistics["objects"] += 1

            if annotation_type == "box":
                statistics["existing_boxes"] += 1
            elif annotation_type == "polygon":
                statistics["converted_polygons"] += 1

    if not output_lines:
        statistics["empty_label_files"] = 1

    output_label_path.parent.mkdir(parents=True, exist_ok=True)

    with output_label_path.open("w", encoding="utf-8") as output_file:
        if output_lines:
            output_file.write("\n".join(output_lines) + "\n")

    return statistics


def assign_groups_to_splits(records: list[dict]) -> dict[str, list[dict]]:
    """
    Split records into train, validation and test subsets.

    Records sharing the same source group key are kept together.
    """
    records_by_group = defaultdict(list)

    for record in records:
        records_by_group[record["group_key"]].append(record)

    groups = list(records_by_group.values())

    random_generator = random.Random(RANDOM_SEED)
    random_generator.shuffle(groups)

    total_records = len(records)

    target_counts = {
        "train": round(total_records * TRAIN_RATIO),
        "valid": round(total_records * VALID_RATIO),
    }

    target_counts["test"] = (
        total_records
        - target_counts["train"]
        - target_counts["valid"]
    )

    assigned_records = {
        "train": [],
        "valid": [],
        "test": [],
    }

    for group in groups:
        remaining_capacity = {
            split_name: target_counts[split_name]
            - len(assigned_records[split_name])
            for split_name in OUTPUT_SPLITS
        }

        eligible_splits = [
            split_name
            for split_name in OUTPUT_SPLITS
            if remaining_capacity[split_name] >= len(group)
        ]

        if eligible_splits:
            selected_split = max(
                eligible_splits,
                key=lambda split_name: remaining_capacity[split_name],
            )
        else:
            selected_split = max(
                OUTPUT_SPLITS,
                key=lambda split_name: remaining_capacity[split_name],
            )

        assigned_records[selected_split].extend(group)

    return assigned_records


def write_data_yaml() -> Path:
    """Write dataset configuration used by Ultralytics training."""
    yaml_path = OUTPUT_DATASET_DIR / "data.yaml"

    yaml_content = (
        f"path: {OUTPUT_DATASET_DIR.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )

    with yaml_path.open("w", encoding="utf-8") as yaml_file:
        yaml_file.write(yaml_content)

    return yaml_path


def write_manifest(
    assigned_records: dict[str, list[dict]],
) -> None:
    """Save a local-only record of the created split."""
    manifest_path = OUTPUT_DATASET_DIR / "split_manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "output_split",
                "image_name",
                "source_split",
                "source_group",
            ],
        )

        writer.writeheader()

        for output_split in OUTPUT_SPLITS:
            for record in assigned_records[output_split]:
                writer.writerow(
                    {
                        "output_split": output_split,
                        "image_name": record["image_path"].name,
                        "source_split": record["source_split"],
                        "source_group": record["group_key"],
                    }
                )


def prepare_output_directory() -> None:
    """Remove a previous prepared dataset and create clean output folders."""
    if OUTPUT_DATASET_DIR.exists():
        shutil.rmtree(OUTPUT_DATASET_DIR)

    for split_name in OUTPUT_SPLITS:
        (OUTPUT_DATASET_DIR / split_name / "images").mkdir(
            parents=True,
            exist_ok=True,
        )

        (OUTPUT_DATASET_DIR / split_name / "labels").mkdir(
            parents=True,
            exist_ok=True,
        )


def main() -> None:
    print("=== Prepare YOLO Detection Dataset ===")
    print(f"Source dataset:   {SOURCE_DATASET_DIR}")
    print(f"Output dataset:   {OUTPUT_DATASET_DIR}")
    print(f"Random seed:      {RANDOM_SEED}")
    print("Split ratio:      train=70%, valid=15%, test=15%\n")

    if not SOURCE_DATASET_DIR.exists():
        print(f"[ERROR] Source dataset not found: {SOURCE_DATASET_DIR}")
        sys.exit(1)

    try:
        records = read_source_records()
        assigned_records = assign_groups_to_splits(records)

        prepare_output_directory()

        split_statistics = {}

        for output_split in OUTPUT_SPLITS:
            statistics = {
                "images": 0,
                "objects": 0,
                "existing_boxes": 0,
                "converted_polygons": 0,
                "empty_label_files": 0,
            }

            for record in assigned_records[output_split]:
                source_image_path = record["image_path"]
                source_label_path = record["label_path"]

                output_image_path = (
                    OUTPUT_DATASET_DIR
                    / output_split
                    / "images"
                    / source_image_path.name
                )

                output_label_path = (
                    OUTPUT_DATASET_DIR
                    / output_split
                    / "labels"
                    / f"{source_image_path.stem}.txt"
                )

                shutil.copy2(source_image_path, output_image_path)

                label_statistics = convert_label_file(
                    source_label_path,
                    output_label_path,
                )

                statistics["images"] += 1

                for key in [
                    "objects",
                    "existing_boxes",
                    "converted_polygons",
                    "empty_label_files",
                ]:
                    statistics[key] += label_statistics[key]

            split_statistics[output_split] = statistics

        yaml_path = write_data_yaml()
        write_manifest(assigned_records)

        total_statistics = {
            key: sum(
                split_statistics[split_name][key]
                for split_name in OUTPUT_SPLITS
            )
            for key in [
                "images",
                "objects",
                "existing_boxes",
                "converted_polygons",
                "empty_label_files",
            ]
        }

        safe_summary = {
            "source_dataset": "License Plates of Vehicles in Turkey",
            "prepared_task": "license_plate_object_detection",
            "class_names": [CLASS_NAME],
            "random_seed": RANDOM_SEED,
            "split_ratio": {
                "train": TRAIN_RATIO,
                "valid": VALID_RATIO,
                "test": TEST_RATIO,
            },
            "conversion": {
                "existing_bounding_boxes_preserved": True,
                "polygon_annotations_converted_to_bounding_boxes": True,
            },
            "splits": split_statistics,
            "overall": total_statistics,
        }

        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

        with SUMMARY_PATH.open("w", encoding="utf-8") as summary_file:
            json.dump(safe_summary, summary_file, indent=2)

        print(
            "Split   | Images | Objects | Existing Boxes | "
            "Converted Polygons | Empty Labels"
        )
        print("-" * 80)

        for split_name in OUTPUT_SPLITS:
            row = split_statistics[split_name]

            print(
                f"{split_name:<7} | "
                f"{row['images']:>6} | "
                f"{row['objects']:>7} | "
                f"{row['existing_boxes']:>14} | "
                f"{row['converted_polygons']:>18} | "
                f"{row['empty_label_files']:>12}"
            )

        print("\n=== Prepared Dataset Summary ===")
        print(f"Total images:                  {total_statistics['images']}")
        print(f"Total objects:                 {total_statistics['objects']}")
        print(f"Existing boxes preserved:      {total_statistics['existing_boxes']}")
        print(f"Polygons converted to boxes:   {total_statistics['converted_polygons']}")
        print(f"Empty label files retained:    {total_statistics['empty_label_files']}")
        print(f"YOLO data configuration:       {yaml_path}")
        print(f"Safe summary saved to:         {SUMMARY_PATH}")

        print("\nPrepared YOLO detection dataset created successfully.")

    except (FileNotFoundError, ValueError) as error:
        print(f"[ERROR] {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()