from ultralytics import YOLO
import os

# Load the weights from previous successful run
weights_path = r'D:/deutschland/BSBI/course content/computer vision/PyCharm projects/project1/runs/segment/Brain_Tumor_V4/weights/last.pt'
model = YOLO(weights_path)

# Define the dataset configuration path
data_yaml_path = r'D:/deutschland/BSBI/course content/computer vision/PyCharm projects/PythonProject/Brain_tumor_Seg_Hy.V4/data.yaml'

# Start the fine-tuning process
if __name__ == '__main__':
    model.train(
        data=data_yaml_path,
        epochs=30,           # Another 30 epochs for a total of 60
        imgsz=640,           # Standard image size for YOLOv8
        device='cpu',
        workers=0,           # Zero to prevent multi-processing access violations
                                                                                                                                                                batch=4,             # Small batch size to keep RAM usage stable
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        cache=False,         # Disable caching to save system memory
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                name='Brain_Tumor_V5_FineTune' # New folder name for the improved results
    )

print("Training initiated successfully. Keep the laptop connected to power.")