import cv2
import json
import os
import numpy as np

# Path to the directory containing images and the COCO JSON file
img_dir = "/Users/bhavish/rbc_plt_annotations/project_1-at-2025-03-25-04-44-bbbea42d/images"   
json_file = "/Users/bhavish/rbc_plt_annotations/project_1-at-2025-03-25-04-44-bbbea42d/result.json"

# Load the COCO JSON data
with open(json_file, "r") as f:
    coco_data = json.load(f)

# Define a color map for different categories
category_colors = {
    0: (0, 255, 255),     # Yellow for "plt"
    1: (255, 0, 0),     # Blue for "plt-clump"
    2: (0, 0, 255),      # Red for "rbc"
    3: (0,255,0)         # green for "wbc"
}

# Create a dictionary mapping category ID to name
category_names = {category["id"]: category["name"] for category in coco_data["categories"]}

# Function to visualize annotations on images
def visualize_annotations(image, annotations):
    for annotation in annotations:
        bbox = annotation["bbox"]
        category_id = annotation["category_id"]
        
        # Get the color and label for the category
        color = category_colors.get(category_id, (255, 255, 255))  # Default to white if not found
        label = category_names.get(category_id, "Unknown")

        # Draw bounding box
        bbox_start = (int(bbox[0]), int(bbox[1]))
        bbox_end = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(image, bbox_start, bbox_end, color, 2)

        # Put category label text
        text_position = (bbox_start[0], bbox_start[1] - 5)
        cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
    return image

# Verify annotations on selected images
for image_data in coco_data["images"]:
    image_id = image_data["id"]
    file_name = image_data["file_name"].split("/")[-1]
    image_path = os.path.join(img_dir, file_name)
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Get annotations for the current image
    annotations = [anno for anno in coco_data["annotations"] if anno["image_id"] == image_id]
    
    # Visualize annotations on the image
    annotated_image = visualize_annotations(image.copy(), annotations)
    
    # Display the annotated image
    cv2.imshow("Annotated Image", annotated_image)
    cv2.waitKey(0)  # Wait for a key press to close the image window

cv2.destroyAllWindows()
