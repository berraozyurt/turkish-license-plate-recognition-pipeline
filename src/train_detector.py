from argparse import ArgumentParser
from pathlib import Path
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

PRIVATE_RUNS_DIR = (
    PROJECT_ROOT
    / "results"
    / "private"
    / "training_runs"
)

SUPPORTED_MODELS = {
    "yolo26n": "yolo26n.pt",
    "yolo11n": "yolo11n.pt",
    "yolo26s": "yolo26s.pt",
}

DEFAULT_SEED = 42


def parse_arguments():
    parser = ArgumentParser(
        description="Train a YOLO detector for Turkish license plate detection."
    )

    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS.keys(),
        default="yolo26n",
        help="Ultralytics YOLO model variant to train.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs for a full experiment.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size. Start with 4 for a 4 GB laptop GPU.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of data loading workers. Zero is safer on Windows.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short one-epoch training test on a small subset.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom name for the training run.",
    )

    return parser.parse_args()


def validate_environment() -> None:
    if not DATA_YAML_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset configuration not found: {DATA_YAML_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was not detected. Training is configured for GPU use."
        )


def get_training_configuration(arguments) -> dict:
    if arguments.smoke_test:
        epochs = 1
        fraction = 0.05
        run_name = arguments.run_name or f"{arguments.model}_smoke_test"
    else:
        epochs = arguments.epochs
        fraction = 1.0
        run_name = arguments.run_name or f"{arguments.model}_training"

    return {
        "epochs": epochs,
        "fraction": fraction,
        "run_name": run_name,
    }


def main() -> None:
    arguments = parse_arguments()

    try:
        validate_environment()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"[ERROR] {error}")
        sys.exit(1)

    configuration = get_training_configuration(arguments)
    model_weights = SUPPORTED_MODELS[arguments.model]

    PRIVATE_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== YOLO License Plate Detector Training ===")
    print(f"Model:           {model_weights}")
    print(f"Dataset config:  {DATA_YAML_PATH}")
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"Epochs:          {configuration['epochs']}")
    print(f"Image size:      {arguments.imgsz}")
    print(f"Batch size:      {arguments.batch}")
    print(f"Training subset: {configuration['fraction']:.2f}")
    print(f"Run name:        {configuration['run_name']}")
    print(f"Output folder:   {PRIVATE_RUNS_DIR}")
    print(f"Smoke test:      {arguments.smoke_test}\n")

    model = YOLO(model_weights)

    model.train(
        data=str(DATA_YAML_PATH),
        epochs=configuration["epochs"],
        imgsz=arguments.imgsz,
        batch=arguments.batch,
        device=0,
        workers=arguments.workers,
        project=str(PRIVATE_RUNS_DIR),
        name=configuration["run_name"],
        exist_ok=arguments.smoke_test,
        optimizer="auto",
        seed=DEFAULT_SEED,
        deterministic=True,
        amp=True,
        cache=False,
        plots=True,
        save=True,
        fraction=configuration["fraction"],
        verbose=True,
    )

    output_path = PRIVATE_RUNS_DIR / configuration["run_name"]

    print("\nTraining process finished.")
    print(f"Local training outputs saved to: {output_path}")
    print("These outputs remain private and are excluded from Git.")


if __name__ == "__main__":
    main()