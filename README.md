# Turkish License Plate Recognition Pipeline

This project implements an independent computer vision pipeline for Turkish license plate detection and OCR-based recognition.

The pipeline combines:

* **YOLO-based license plate detection**
* **Detection model comparison**
* **Plate crop extraction**
* **OCR-based text recognition**
* **Privacy-safe reporting**

No company code, internship data, private vehicle images, readable real plate identifiers or trained model weights are included in this repository.

## Project Overview

The project is built around the following workflow:

```text
Input Vehicle Image
- License Plate Detection with YOLO
- Plate Crop Extraction
- OCR with EasyOCR
- Turkish Plate Format Normalization
- Privacy-Safe Metrics
```

The detection component is trained and evaluated on a public Turkish license plate detection dataset. The OCR component is evaluated quantitatively using synthetically generated Turkish plate images to avoid publishing or transcribing real plate identifiers.

## Motivation

Automatic License Plate Recognition (ALPR) is a common computer vision problem used in traffic monitoring, access control and vehicle management systems.

This project was developed as a portfolio implementation after prior internship experience with YOLO-based license plate workflows. The original internship project and data were company-related, so this repository was built independently using public and synthetic resources.

## Dataset

The detection component uses the **License Plates of Vehicles in Turkey** dataset published on Roboflow Universe.

* Task: License plate object detection
* Source images: 3,500
* Annotated license plate objects: 6,320
* License: CC BY 4.0
* Source: [License Plates of Vehicles in Turkey — Roboflow Universe](https://universe.roboflow.com/tr-plaka-recognition/license-plates-of-vehicles-in-turkey-s3tbj-s5lcc/dataset/1)

The dataset is not redistributed in this repository.

## Dataset Preparation

The downloaded dataset contains mixed annotation geometry:

| Annotation Type              |     Count |
| ---------------------------- | --------: |
| Existing YOLO bounding boxes |       107 |
| Polygon annotations          |     6,213 |
| **Total annotated objects**  | **6,320** |

Since the detector is trained as an object detection model, polygon annotations are automatically converted into enclosing YOLO bounding boxes.

The original downloaded split had a very small test partition. Therefore, this project creates a new deterministic split using a fixed random seed while keeping files derived from the same source image in the same subset.

| Split      |    Images | License Plate Objects |
| ---------- | --------: | --------------------: |
| Train      |     2,450 |                 4,391 |
| Validation |       525 |                   984 |
| Test       |       525 |                   945 |
| **Total**  | **3,500** |             **6,320** |

Dataset preparation and validation scripts:

```bash
python src/check_dataset.py
python src/prepare_detection_dataset.py
python src/validate_prepared_dataset.py
```

The prepared dataset is generated locally under:

```text
data/prepared_dataset/
```

This folder is excluded from Git.

## Detection Model Experiments

Two lightweight Ultralytics YOLO detector variants were trained under the same experimental conditions:

* YOLO26n
* YOLO11n

Both models were trained using:

| Setting       | Value                                     |
| ------------- | ----------------------------------------- |
| Epochs        | 50                                        |
| Image size    | 640                                       |
| Batch size    | 4                                         |
| Random seed   | 42                                        |
| Device        | NVIDIA GPU                                |
| Dataset split | Same prepared train/validation/test split |

Training command example:

```bash
python src/train_detector.py --model yolo26n --full-train --epochs 50
```

The training outputs, model weights and prediction images are saved locally under:

```text
results/private/
```

They are excluded from Git because they may contain real vehicle images or readable plate information.

## Validation Results

Model selection was performed using the validation set only.

| Model       |  Precision |     Recall |      mAP50 |   mAP50-95 |       Inference Time |
| ----------- | ---------: | ---------: | ---------: | ---------: | -------------------: |
| **YOLO26n** | **0.9060** | **0.9506** | **0.9715** | **0.7825** | **10.1043 ms/image** |
| YOLO11n     |     0.8872 |     0.9472 |     0.9642 |     0.7627 |     10.4661 ms/image |

`YOLO26n` was selected as the final detector because it achieved stronger validation performance, especially on mAP50-95.

## Final Detection Test Results

After selecting the final detector using validation results, `YOLO26n` was evaluated once on the held-out test set.

| Final Detector | Split | Images | Objects |  Precision |     Recall |      mAP50 |   mAP50-95 |      Inference Time |
| -------------- | ----- | -----: | ------: | ---------: | ---------: | ---------: | ---------: | ------------------: |
| **YOLO26n**    | Test  |    525 |     945 | **0.9046** | **0.9428** | **0.9735** | **0.7809** | **9.9814 ms/image** |

This result is close to the validation result, which suggests that the selected detector generalizes well to the held-out test split.

Evaluation command:

```bash
python src/evaluate_detector.py --experiment yolo26n --weights "results/private/training_runs/yolo26n_training/weights/best.pt" --split test --confirm-final-test
```

## OCR Component

The OCR stage uses EasyOCR to read cropped license plate regions.

For privacy reasons, real license plate text from real vehicle images is not included in public result files. Instead, OCR is quantitatively evaluated on synthetically generated Turkish-style license plate images.

Synthetic OCR dataset generation:

```bash
python src/generate_synthetic_plates.py
```

OCR evaluation:

```bash
python src/evaluate_synthetic_ocr.py
```

## Synthetic OCR Results

| OCR Metric                      |     Result |
| ------------------------------- | ---------: |
| Samples                         |        300 |
| Exact match accuracy            | **0.9633** |
| Mean character accuracy         | **0.9889** |
| Valid Turkish plate format rate | **0.9667** |

These OCR metrics are calculated on synthetic plate images generated by the project. They do not contain real vehicle plate identifiers.

## Full Pipeline Demo

The full detection + OCR pipeline can be run locally on a limited number of real test images:

```bash
python src/run_full_pipeline.py --limit 30 --conf 0.40
```

The local full pipeline processes real test images as follows:

```text
YOLO26n detection
→ plate crop extraction
→ EasyOCR recognition
→ Turkish plate format check
```

For privacy, readable OCR predictions from real plate crops are saved only under:

```text
results/private/
```

The public summary contains only aggregate values.

Example local full pipeline summary on 30 test images:

| Metric                             |  Value |
| ---------------------------------- | -----: |
| Images processed                   |     30 |
| Detections                         |     44 |
| Images with detection              |     28 |
| Mean detections per image          | 1.4667 |
| Valid plate-format OCR output rate | 0.4318 |

This value is not treated as OCR accuracy because real test images do not include public ground-truth plate transcriptions. It only measures how often the OCR output matches a Turkish plate-like format.

## Project Structure

```text
turkish-license-plate-recognition-pipeline/
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── results/
│   ├── README.md
│   └── metrics/
│       ├── detection/
│       ├── ocr/
│       └── pipeline/
│
├── src/
│   ├── check_dataset.py
│   ├── check_environment.py
│   ├── check_ocr_environment.py
│   ├── evaluate_detector.py
│   ├── evaluate_synthetic_ocr.py
│   ├── generate_synthetic_plates.py
│   ├── prepare_detection_dataset.py
│   ├── run_full_pipeline.py
│   ├── train_detector.py
│   └── validate_prepared_dataset.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

A separate Python environment is recommended.

```bash
conda create -n plate_alpr python=3.11 -y
```

Install dependencies:

```bash
pip install -r requirements.txt
```

This project was tested with:

| Package     | Version           |
| ----------- | ----------------- |
| Python      | 3.11.15           |
| PyTorch     | 2.5.1 + CUDA 12.1 |
| Ultralytics | 8.4.56            |
| EasyOCR     | 1.7.2             |
| OpenCV      | 4.13.0            |
| NumPy       | 2.4.4             |
| Pandas      | 3.0.3             |

Environment validation:

```bash
python src/check_environment.py
python src/check_ocr_environment.py
```

## Usage

- 1. Validate the downloaded detection dataset
python src/check_dataset.py

- 2. Prepare the YOLO detection dataset
python src/prepare_detection_dataset.py

- 3. Validate the prepared dataset
python src/validate_prepared_dataset.py

- 4. Train a detector
Short smoke test:
python src/train_detector.py --model yolo26n

Full training:

python src/train_detector.py --model yolo26n --full-train --epochs 50

- 5. Evaluate a trained detector

Validation evaluation:

python src/evaluate_detector.py --experiment yolo26n --weights "results/private/training_runs/yolo26n_training/weights/best.pt"

Final test evaluation:

python src/evaluate_detector.py --experiment yolo26n --weights "results/private/training_runs/yolo26n_training/weights/best.pt" --split test --confirm-final-test


- 6. Evaluate OCR on synthetic plates

python src/generate_synthetic_plates.py
python src/evaluate_synthetic_ocr.py

- 7. Run the full local pipeline
python src/run_full_pipeline.py --limit 30 --conf 0.40


## Privacy Notes

This repository intentionally excludes:

* real vehicle images,
* license plate crop images,
* readable real plate OCR predictions,
* dataset annotation files,
* trained model weights,
* private training outputs.

The public repository contains only code, documentation and privacy-safe aggregate metrics.

## Limitations

* The detector is trained on a public dataset and may not generalize to all real-world camera angles, lighting conditions or plate styles.
* OCR accuracy on real plate crops is not publicly reported because real plate text ground truth is not included.
* The OCR component is evaluated quantitatively using synthetic plate images.
* The full pipeline demo reports format-valid OCR output rate, not true OCR accuracy.
* Trained weights are not included in the repository.

## Project Status

Completed:

* Dataset validation
* Polygon-to-bounding-box conversion
* Deterministic train/validation/test split
* YOLO26n detector training
* YOLO11n comparison experiment
* Final YOLO26n test evaluation
* Synthetic OCR evaluation
* Full local detection + OCR pipeline

Planned improvements:

* Add optional anonymized visual examples
* Improve OCR preprocessing for real plate crops
* Compare additional OCR engines
* Export a lightweight inference-only script
