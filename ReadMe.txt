# Brain Tumor Hybrid Localization Framework

This repository contains a **Hybrid AI Framework** for intracranial pathology segmentation and spatial landmark extraction. Using **YOLOv8-seg** and a custom **Geometric Logic**, the system identifies tumors and extracts 3D-ready coordinates (Top, Center, Bottom).

##  Key Features
- **Instance Segmentation:** High-precision masks for Glioma, Meningioma, and Pituitary tumors.
- **Hybrid Localization:** Automated extraction of anatomical landmarks ($P_{top}, P_{avg}, P_{bottom}$).
- **CPU Optimized:** Designed to run efficiently on standard hardware.
- **Data Export:** Direct export of spatial coordinates to CSV/Excel for clinical analysis.

## Stack
- **Deep Learning:** YOLOv8 (Ultralytics)
- **Annotation:** CVAT.AI
- **Languages:** Python 3.14
- **Libraries:** OpenCV, NumPy, Matplotlib, Pandas

## Results
The model achieved an **mAP@50 of 0.588** with 100% accuracy in identifying healthy (No-Tumor) cases.

## Detection & Localization Results
| Pituitary | Meningioma | Glioma |
| :---: | :---: | :---: |
| ![Pituitary](./results/pit.jpg) | ![Meningioma](./results/men.jpg) | ![Glioma](./results/gli.jpg) |

## Performance Metrics
| Confusion Matrix | F1-Confidence Curve |
| :---: | :---: |
| ![Confusion Matrix](./results/confusion_matrix_normalized.png) | ![F1 Curve](./results/BoxF1_curve.png) |

## Project Structure
- `/script`: Contains `train.py`, `test.py`, and the hybrid extraction logic.
- `/models`: Contains the trained weights (`best.pt`).

- `/results`: Performance curves and confusion matrix.






