# Dataset Setup

The dataset files used in this project are stored locally and are intentionally excluded from version control.

## Detection Dataset

The detection component uses the **License Plates of Vehicles in Turkey** dataset published on Roboflow Universe.

* Task: license plate object detection
* Source dataset size: 3,500 images
* Source annotations: 6,320 annotated license plate objects
* License: CC BY 4.0
* Source: [License Plates of Vehicles in Turkey — Roboflow Universe](https://universe.roboflow.com/tr-plaka-recognition/license-plates-of-vehicles-in-turkey-s3tbj-s5lcc/dataset/1)

## Annotation Preparation

The downloaded export contains mixed annotation geometry:

* 107 annotations already provided as YOLO bounding boxes
* 6,213 annotations provided as normalized polygon outlines

Because this project trains an object detection model, polygon annotations are converted automatically into enclosing YOLO bounding boxes by:

```bash
python src/prepare_detection_dataset.py
```

No manual relabeling is required.

## Prepared Detection Split

The original downloaded split contains only a very small test partition. Therefore, the project creates a new deterministic split using a fixed random seed while keeping files derived from the same source image in the same subset.

| Split      |    Images | License Plate Objects |
| ---------- | --------: | --------------------: |
| Train      |     2,450 |                 4,391 |
| Validation |       525 |                   984 |
| Test       |       525 |                   945 |
| **Total**  | **3,500** |             **6,320** |

The prepared dataset is generated locally under:

```text
data/prepared_dataset/
```

## Local Directory Structure

After downloading the dataset, place the source export in:

```text
data/
├── README.md
├── detection_dataset/
│   ├── data.yaml
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
└── prepared_dataset/        # Generated locally by the preparation script
    ├── data.yaml
    ├── train/
    ├── valid/
    └── test/
```

## Privacy and Redistribution

Real vehicle images, readable license plates, annotation files and prepared local dataset contents are not included in the public repository.

Only privacy-safe aggregate dataset summaries, code and later evaluation metrics are intended for publication.
