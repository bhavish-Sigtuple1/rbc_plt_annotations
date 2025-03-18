import cv2
import os
import random
import numpy as np
import json

img_path = '/Users/bhavish/Downloads/rbc_plt_annotations/Sigvet_rbc_data_Recon/rbc_image_test_input_folder'  
dst_path = "/Users/sasikiranp/Desktop/SigVet_RBC_Anno/Data/Canine2/53194/adaptive_results"


os.makedirs(dst_path, exist_ok=True)

files = os.listdir(img_path)
coco_data = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "rbc"}],
}

annotation_id = 0


def apply_adaptive_threshold(image):
    adaptive_thresh = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 
        3   
    )
    return adaptive_thresh

def remove_small_objects(binary_image, min_size):
    inverted_image = cv2.bitwise_not(binary_image)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_image, connectivity=8)
    filtered_image = np.zeros(binary_image.shape, dtype=np.uint8)
    for i in range(1, num_labels): 
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_image[labels == i] = 255
    filtered_image = cv2.bitwise_not(filtered_image)
    return filtered_image


def fill_inner_holes(binary_image):
 
    inverted_image = cv2.bitwise_not(binary_image)
    
    # Perform morphological closing to fill holes
    kernel = np.ones((3,3), np.uint8)  # Adjust kernel size as needed
    closed_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)
    
    # Invert the result back
    filled_image = cv2.bitwise_not(closed_image)

    return filled_image


def segment_image(image):
    if image is None:
        print(f"Error: Could not load image at {image}. Please check the file path.")
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = apply_adaptive_threshold(gray_image)
    min_object_size = 500  
    cleaned_image = remove_small_objects(binary_image, min_object_size)

    filled_image = fill_inner_holes(cleaned_image)
    return filled_image

for idx, file in enumerate(files):
    # Skip hidden files or non-image files
    if file.startswith('.') or not file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
        continue
    
    image = cv2.imread(os.path.join(img_path, file))
    
    # Check if image is loaded correctly
    if image is None:
        print(f"Warning: Could not load image at {os.path.join(img_path, file)}. Skipping this file.")
        continue
    
    img = np.zeros(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_1 = gray.copy()
    th1 = segment_image(image)

    # Find contours
    contours, _ = cv2.findContours(~th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        w = cv2.boundingRect(contour)[2]
        h = cv2.boundingRect(contour)[3]

        if w * h > 4000:
            continue
        annotation = {
            "id": annotation_id,
            "image_id": idx,  # Index of the current image in COCO data
            "category_id": 1,  # Assuming category id for 'segment'
            "area": w * h,
            "bbox": cv2.boundingRect(contour),
            "iscrowd": 0,
        }

        coco_data["annotations"].append(annotation)
        annotation_id += 1
        # Draw bounding box (optional)
        bbox_start = (int(bbox[0]), int(bbox[1]))
        bbox_end = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(image_1, bbox_start, bbox_end, (0, 255, 0), 1)

    cv2.imwrite(os.path.join(dst_path, file), image_1)

    # Save image entry
    coco_data["images"].append({"id": idx, "file_name": file})

# Save COCO JSON file
with open("/Users/sasikiranp/Desktop/SigVet_RBC_Anno/Data/Canine2/53194/coco_json_adaptive.json", "w") as f:
    json.dump(coco_data, f, indent=4)
