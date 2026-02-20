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
| ![Pituitary](./results/Result_PITUITARY-TRAIN_438_jpg.rf.180e90f5701de1061c33e0480865da41.jpg) | ![Meningioma](./results/Result_MENINGIOMA-VALID_150_jpg.rf.040f2998a4af373dff26acbd00db2230.jpg) | ![Glioma](./results/Result_GLIOMA-TRAIN_124_jpg.rf.a1cbc288a62c0d88984dd44466f173a4.jpg) |

## Performance Metrics
| Confusion Matrix | F1-Confidence Curve |
| :---: | :---: |
| ![Confusion Matrix](./results/confusion_matrix_normalized.png) | ![F1 Curve](./results/BoxF1_curve.png) |

## Project Structure
- `/script`: Contains `train.py`, `test.py`, and the hybrid extraction logic.
- `/models`: Contains the trained weights (`best.pt`).

- `/results`: Performance curves and confusion matrix.





