from pathlib import Path
import csv
import random
import string

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "ocr_dataset"
    / "synthetic_plates"
)

IMAGES_DIR = OUTPUT_DIR / "images"
MANIFEST_PATH = OUTPUT_DIR / "synthetic_plate_manifest.csv"

RANDOM_SEED = 42
SAMPLE_COUNT = 300

IMAGE_WIDTH = 360
IMAGE_HEIGHT = 90

LETTERS = [
    letter
    for letter in string.ascii_uppercase
    if letter not in {"Q", "W", "X"}
]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]

    for font_path in font_candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)

    return ImageFont.load_default()


def generate_plate_text(random_generator: random.Random) -> str:
    """
    Generate a synthetic Turkish-style license plate text.

    Examples:
    34 ABC 123
    06 AB 4821
    35 A 7214
    """
    province_code = random_generator.randint(1, 81)
    letter_count = random_generator.choice([1, 2, 3])

    if letter_count == 1:
        digit_count = random_generator.choice([3, 4])
    elif letter_count == 2:
        digit_count = random_generator.choice([3, 4])
    else:
        digit_count = random_generator.choice([2, 3])

    letters = "".join(
        random_generator.choice(LETTERS)
        for _ in range(letter_count)
    )

    digits = "".join(
        random_generator.choice(string.digits)
        for _ in range(digit_count)
    )

    return f"{province_code:02d} {letters} {digits}"


def draw_synthetic_plate(
    plate_text: str,
    random_generator: random.Random,
) -> np.ndarray:
    background = Image.new(
        "RGB",
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        color=(245, 245, 240),
    )

    draw = ImageDraw.Draw(background)

    # Border
    border_margin = 5
    draw.rounded_rectangle(
        [
            border_margin,
            border_margin,
            IMAGE_WIDTH - border_margin,
            IMAGE_HEIGHT - border_margin,
        ],
        radius=8,
        outline=(20, 20, 20),
        width=3,
    )

    # Small blue strip similar to European-style plates
    strip_width = 42
    draw.rectangle(
        [border_margin, border_margin, strip_width, IMAGE_HEIGHT - border_margin],
        fill=(30, 80, 180),
    )

    small_font = load_font(16)
    draw.text(
        (13, 56),
        "TR",
        fill=(255, 255, 255),
        font=small_font,
    )

    font = load_font(46)

    text_bbox = draw.textbbox((0, 0), plate_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x = strip_width + (IMAGE_WIDTH - strip_width - text_width) // 2
    y = (IMAGE_HEIGHT - text_height) // 2 - 3

    draw.text(
        (x, y),
        plate_text,
        fill=(10, 10, 10),
        font=font,
    )

    image = np.array(background)

    brightness_shift = random_generator.randint(-10, 10)
    image = np.clip(
        image.astype(np.int16) + brightness_shift,
        0,
        255,
    ).astype(np.uint8)

    if random_generator.random() < 0.35:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    noise = random_generator.normalvariate(0, 1.5)
    if abs(noise) > 0:
        noise_array = np.random.normal(
            loc=0,
            scale=1.5,
            size=image.shape,
        )
        image = np.clip(
            image.astype(np.float32) + noise_array,
            0,
            255,
        ).astype(np.uint8)

    return image


def normalize_plate_text(text: str) -> str:

    return "".join(text.upper().split())


def main() -> None:
    print(" Generate Synthetic Turkish Plate OCR Dataset ")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Sample count:     {SAMPLE_COUNT}")

    random_generator = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for index in range(SAMPLE_COUNT):
        plate_text = generate_plate_text(random_generator)
        image = draw_synthetic_plate(plate_text, random_generator)

        file_name = f"synthetic_plate_{index:04d}.png"
        image_path = IMAGES_DIR / file_name

        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        rows.append(
            {
                "file_name": file_name,
                "plate_text": plate_text,
                "normalized_plate_text": normalize_plate_text(plate_text),
            }
        )

    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "file_name",
                "plate_text",
                "normalized_plate_text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSynthetic OCR dataset generated successfully.")
    print(f"Images:   {IMAGES_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    print("These files are local and excluded from Git.")


if __name__ == "__main__":
    main()