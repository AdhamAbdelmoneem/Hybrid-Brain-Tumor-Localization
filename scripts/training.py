from torch.fx.experimental.partitioner_utils import Device
from ultralytics import YOLO

# Loading the segmentation model
model = YOLO('yolov8n-seg.pt')

# Execute the final training on the balanced dataset
# Training for 30 epochs with 640px resolution for high precision
results = model.train(
    data="Brain_tumor_Seg_Hy.V4/data.yaml",
    epochs=30,
    imgsz=640,
    batch=8,
    name='Brain_Tumor_V4',
    device='cpu',
    # Adding Augmentation Paras for better Accuracy
    degrees=15.0,    # Rotate image by +/- 15 degrees
    translate=0.1,   # Translate image horizontally/vertically by 10%
    scale=0.5,       # Scale image by +/- 50% (helps with different sizes)
    shear=2.0,       # Shear image by +/- 2 degrees
    flipud=0.0,      # Vertical flip is not common in MRI, so we kept it 0
    fliplr=0.5,      # Horizontal flip (Left to Right and Opposite)
    mosaic=1.0,      # Combine 4 images into one to help detect small tumors
    mixup=0.1        # Mix two images to improve model generalization
)

print("Training Complete! Check the results.")