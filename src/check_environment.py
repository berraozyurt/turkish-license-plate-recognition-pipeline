from pathlib import Path
import sys


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def check_python() -> None:
    print_section("Python")
    print(f"Executable: {sys.executable}")
    print(f"Version:    {sys.version.split()[0]}")


def check_torch() -> None:
    print_section("PyTorch and CUDA")

    try:
        import torch
    except ImportError:
        print("[ERROR] torch is not installed.")
        return

    print(f"Torch version:      {torch.__version__}")
    print(f"CUDA build:         {torch.version.cuda}")
    print(f"CUDA available:     {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU device:         {torch.cuda.get_device_name(0)}")
        print(f"GPU count:          {torch.cuda.device_count()}")


def check_ultralytics() -> None:
    print_section("Ultralytics YOLO")

    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics is not installed.")
        return

    print(f"Ultralytics version: {ultralytics.__version__}")

    try:
        model = YOLO("yolo26n.pt")
        print("YOLO26n load test:   OK")
        print(f"Model task:          {model.task}")
    except Exception as error:
        print(f"[ERROR] YOLO26n load test failed: {error}")


def check_common_libraries() -> None:
    print_section("Common Libraries")

    libraries = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("yaml", "pyyaml"),
    ]

    for import_name, package_name in libraries:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "version not exposed")
            print(f"{package_name:<15}: {version}")
        except ImportError:
            print(f"{package_name:<15}: NOT INSTALLED")


def check_project_paths() -> None:
    print_section("Project Paths")

    project_root = Path(__file__).resolve().parents[1]
    prepared_yaml = project_root / "data" / "prepared_dataset" / "data.yaml"

    print(f"Project root:        {project_root}")
    print(f"Prepared data.yaml:  {prepared_yaml}")
    print(f"Data YAML exists:    {prepared_yaml.exists()}")


def main() -> None:
    print("Environment Check for Turkish License Plate Recognition Pipeline")

    check_python()
    check_torch()
    check_ultralytics()
    check_common_libraries()
    check_project_paths()


if __name__ == "__main__":
    main()