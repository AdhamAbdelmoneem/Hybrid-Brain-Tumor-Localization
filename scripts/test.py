from ultralytics import YOLO
import cv2
import numpy as np
import os
import random

# Load the fine-tuned model
model = YOLO(
    r'D:/deutschland/BSBI/course content/computer vision/PyCharm projects/project1/runs/segment/Brain_Tumor_V5_FineTune/weights/best.pt')

# Path to the TEST folder
test_folder = 'Brain_tumor_Seg_Hy.V4/test/images'
output_folder = 'Research_Results_Bulk'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get all images from the test folder
all_images = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Select 10 random images
selected_images = random.sample(all_images, min(10, len(all_images)))

print(f"Starting bulk processing for {len(selected_images)} images...")

for img_name in selected_images:
    img_path = os.path.join(test_folder, img_name)
    results = model.predict(source=img_path, conf=0.25)  # Balanced confidence

    for result in results:
        # Plotting original YOLO mask/box
        img_plot = result.plot(labels=True, boxes=True)

        if result.masks is not None:
            mask_coords = result.masks.xy[0]

            # Hybrid Point Calculations
            cx = int(np.mean(mask_coords[:, 0]))
            cy = int(np.mean(mask_coords[:, 1]))
            top_idx = np.argmin(mask_coords[:, 1])
            top_pt = (int(mask_coords[top_idx, 0]), int(mask_coords[top_idx, 1]))
            bottom_idx = np.argmax(mask_coords[:, 1])
            bottom_pt = (int(mask_coords[bottom_idx, 0]), int(mask_coords[bottom_idx, 1]))

            # Drawing Labels on the Image
            font = cv2.FONT_HERSHEY_DUPLEX
            # Points
            cv2.circle(img_plot, (cx, cy), 5, (0, 255, 0), -1)
            cv2.circle(img_plot, top_pt, 5, (255, 255, 255), -1)
            cv2.circle(img_plot, bottom_pt, 5, (0, 0, 255), -1)

            # Texts
            cv2.putText(img_plot, "Top", (top_pt[0] + 5, top_pt[1] - 5), font, 0.6, (255, 255, 255), 1)
            cv2.putText(img_plot, "Center", (cx + 5, cy - 5), font, 0.6, (0, 255, 0), 1)
            cv2.putText(img_plot, "Bottom", (bottom_pt[0] + 5, bottom_pt[1] + 15), font, 0.6, (0, 0, 255), 1)

        # Save each image in the new folder
        save_path = os.path.join(output_folder, f"Result_{img_name}")
        cv2.imwrite(save_path, img_plot)

print(f"Bulk processing finished. Check the folder: {output_folder}")
