import cv2
import os
import numpy as np


img_path = "/Users/bhavish/Desktop/Data_Recon_RBC_PLT_ANN/06ba4122-b7cc-4a6b-af8e-7c2544988f74/output_816"
des_dir_extractor = "/Users/bhavish/Desktop/Data_Recon_RBC_PLT_ANN/06ba4122-b7cc-4a6b-af8e-7c2544988f74/output_2x_416"     
os.makedirs(des_dir_extractor, exist_ok=True)

# Get all files from img_path
files = os.listdir(img_path)

# Filter out .DS_Store if it's present
files = [f for f in files if f != ".DS_Store"]

for idx, img_name in enumerate(files):
    img_path_full = os.path.join(img_path, img_name)
    
    # Ensure image was successfully loaded
    img = cv2.imread(img_path_full)
    if img is None:
        print(f"Failed to read {img_name}. Skipping...")
        continue

    # Check if the image is 832x832
    if img.shape[0] != 832 or img.shape[1] != 832:
        print(f"Skipping {img_name}, since it is not 832x832.")
        continue

    # Split the 832x832 image into 4 patches of 416x416
    for i in range(2):  # For both rows
        for j in range(2):  # For both columns
            x_start = j * 416
            y_start = i * 416
            patch_416 = img[y_start:y_start + 416, x_start:x_start + 416]

            # Split each 416x416 patch into 4 patches of 208x208
            for p in range(2):  # For both rows of 208x208 patches
                for q in range(2):  # For both columns of 208x208 patches
                    x_start_208 = q * 208
                    y_start_208 = p * 208 
                    patch_208 = patch_416[y_start_208:y_start_208 + 208, x_start_208:x_start_208 + 208]

                    # Apply 2x zoom to the 208x208 patch to make it 416x416
                    patch_416_zoomed = cv2.resize(patch_208, (416, 416), interpolation=cv2.INTER_LINEAR)

                    # Save the final zoomed patch with an appropriate name
                    base_name = os.path.splitext(img_name)[0]
                    patch_name = f"{base_name}_patch_{i*2 + j}_{p*2 + q}.png"
                    save_path = os.path.join(des_dir_extractor, patch_name)
                    cv2.imwrite(save_path, patch_416_zoomed)

    print(f"Processed image {idx + 1}/{len(files)}: {img_name}")
