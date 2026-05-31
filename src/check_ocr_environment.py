import torch


def main() -> None:
    print("=== OCR Environment Check ===")

    try:
        import easyocr
    except ImportError:
        print("[ERROR] easyocr is not installed.")
        return

    print("EasyOCR import: OK")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    print("\nCreating EasyOCR reader...")
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    print("EasyOCR reader: OK")

    print("\nOCR environment is ready.")


if __name__ == "__main__":
    main()