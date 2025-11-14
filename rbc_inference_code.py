import json
import os
import time
import cv2
import numpy as np
import pandas as pd
import csv
from sahi.prediction import ObjectPrediction
from sahi.base import DetectionModel
from yolox_onnx_inference import YOLOX_ONNX
from sahi.predict import get_sliced_prediction
from typing import Any, Dict, List, Optional

# Model and directories
model_path = "/Users/bhavish/rbc_plt_annotations/rbc_plt_annotations/Models/rbc_plt_iter_13_2.onnx"
input_folder = "/Users/bhavish/Desktop/rbc_ns_check"
test_img_dir = f"{input_folder}/output_3b9942f"
dst_path = f"{input_folder}/yolo-results"
output_csv_path = f"{input_folder}/yolo-results.csv"
output_json_path = f"{input_folder}/yolo-results_rbc_plt_coco.json"

# Class and color mappings
class_mapping = {"0": "plt", "1": "plt-clump", "2": "rbc", "3":"rbc-ghost","4":"rbc-nonspherical" ,"5":"wbc"}
# class_mapping = {"0": "plt", "1": "plt-clump", "2": "rbc", "3":"wbc"}

color_mapping = {
    "0": (0, 255, 255),    
    "1": (255, 0, 0),     
    "2": (0, 0, 255),       
    "3": (255,255,0),
    "4": (255,0,255),          
    "5": (0,255,0)
}

# Define class-wise confidence thresholds
class_confidence_thresholds = {
    0: 0.0,  # Threshold for class 'plt'
    1: 0.0,  # Threshold for class 'plt-clump'
    2: 0.0,   # Threshold for class 'rbc'
    3: 0.0, # Threshold for class 'rbc ghost'
    4: 0.3,
    5: 0.0  # Threshold for class 'wbc'
}

# YOLOX ONNX Wrapper Class
class YOLOX_ONNX_SAHI_Wrapper(DetectionModel):
    def __init__(self, model_path, confidence_threshold, category_mapping, load_at_init, image_size):
        self.model_path = model_path
        self.category_mapping = category_mapping
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.load_at_init = load_at_init
        self.num_classes = None
        self.category_remapping = None
        self.load_model()

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes

    def load_model(self):
        model = YOLOX_ONNX(self.model_path)
        self.set_model(model=model)

    def set_model(self, model):
        self.model = model

    @property
    def num_categories(self):
        return len(self.category_mapping)

    @property
    def has_mask(self):
        return False

    @property
    def category_names(self):
        return list(self.category_mapping.values())

    def perform_inference(self, image, image_size=None):
        prediction_result = self.model.infer(image)
        self._original_predictions = prediction_result

    def _create_object_prediction_list_from_original_predictions(self, shift_amount_lists: Optional[List[List[int]]] = [[0, 0]], full_shape_list: Optional[List[List[int]]] = None):
        self._object_prediction_list_per_image = list()
        original_predictions = self._original_predictions
        for index, shift_amount_list in enumerate(shift_amount_lists):
            if isinstance(shift_amount_list[0], int):
                shift_amount_list = [shift_amount_list]
            if full_shape_list is not None and isinstance(full_shape_list[0], int):
                full_shape_list = [full_shape_list]
            object_prediction_list = []
            object_prediction_list_per_image = []
            shift_amount = shift_amount_list[0]
            full_shape = None if full_shape_list is None else full_shape_list[0]
            if original_predictions[index] is not None:
                final_boxes, final_scores, final_classes = original_predictions[index][:, :4], original_predictions[index][:, 4], original_predictions[index][:, 5]
                for i, box in enumerate(final_boxes):
                    object_prediction = ObjectPrediction(
                        bbox=box,
                        bool_mask=None,
                        category_id=int(final_classes[i]),
                        category_name=self.category_mapping[str(int(final_classes[i]))],
                        shift_amount=shift_amount,
                        score=final_scores[i],
                        full_shape=full_shape,
                    )
                    object_prediction_list.append(object_prediction)
                object_prediction_list_per_image.append(object_prediction_list)
                self._object_prediction_list_per_image.append(object_prediction_list_per_image)
            else:
                self._object_prediction_list_per_image.append([[]])

# Function to draw bounding boxes
def draw_bbox(org_img, bbox_start, bbox_end, class_id):
    color = color_mapping.get(str(class_id), (255, 255, 255))  
    cv2.rectangle(org_img, bbox_start, bbox_end, color, 1)

# Initialize model
detection_model = YOLOX_ONNX_SAHI_Wrapper(model_path, 0.6, class_mapping, False, 416)

# Create output directory if not exists
os.makedirs(dst_path, exist_ok=True)

scale = 1 / 1
bbox_aspect_ratio_threshold = 1.75

# Function to save detection counts to CSV
def save_detection_counts_to_csv(image_data, output_csv_path):
    # Define the header for the CSV 
    header = ['Image_Name', 'plt_count', 'plt-clump_count', 'rbc_count','rbc-nonspherical', 'wbc_count']
    
    # Open the CSV file in write mode
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
        # Write the counts for each image
        for img_name, data in image_data.items():
            plt_count = sum(1 for box in data['boxes'] if box[4] == 0)  # Count class 0 (plt)
            plt_clump_count = sum(1 for box in data['boxes'] if box[4] == 1)  # Count class 1 (plt-clump)
            rbc_count = sum(1 for box in data['boxes'] if box[4] == 2)  # Count class 2 (rbc)
            wbc_count = sum(1 for box in data['boxes'] if box[4] == 3)  # Count class 3 (wbc)
            writer.writerow([img_name, plt_count, plt_clump_count, rbc_count, wbc_count])

# Function to save prediction bbox and class to COCO JSON
def save_prediction_to_coco_json(image_data, output_json_path):
    # Prepare COCO JSON structure
    coco_json = {
        "info": {"description": "Generated Annotations"},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "plt", "supercategory": "cell"},
            {"id": 2, "name": "plt-clump", "supercategory": "cell"},
            {"id": 3, "name": "rbc", "supercategory": "cell"},
            {"id": 4, "name": "rbc ghost", "supercategory": "cell"},
            {"id": 5, "name": "rbc-nonspherical","supercategory": "cell"},
            {"id": 6, "name": "wbc", "supercategory": "cell"}
        ],
        "images": [],
        "annotations": []
    }
    
    # Initialize annotation ID
    annotation_id = 1
    image_id = 1
    
    # Process each image's detections
    for img_name, data in image_data.items():
        # Load image to get dimensions
        img_path = os.path.join(test_img_dir, img_name)
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        
        # Add image info to COCO JSON
        coco_json["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })
        
        for box in data['boxes']:
            x, y, width, height, category_id = box
            
            # Create a COCO annotation entry
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id + 1,  # COCO uses 1-based indexing
                "bbox": [x, y, width, height],
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0
            }
            
            coco_json["annotations"].append(annotation)
            annotation_id += 1
        
        image_id += 1
    
    # Save the COCO JSON to a file
    with open(output_json_path, "w") as f:
        json.dump(coco_json, f, indent=4)
    
    print(f"Saved COCO JSON to {output_json_path}")
    print(f"Total annotations: {len(coco_json['annotations'])}")

# Function to extract cells and save results
def extract_cells():
    count_1 = 0
    image_data = {}
    images = []
    box_info = []
    try:
        files = os.listdir(test_img_dir)
        for idx, img_path in enumerate(files):
            img_name = os.path.basename(img_path)
            if img_name == ".DS_Store":
                continue
            img = cv2.imread(os.path.join(test_img_dir, img_path))
            org_img = img.copy()
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

            result = get_sliced_prediction(
                img[:, :, ::-1],
                detection_model,    
                slice_height=416,
                slice_width=416,
                batch_size=1,
                overlap_height_ratio=0.1923,
                overlap_width_ratio=0.1923,
                perform_standard_pred=False,
                verbose=0
            )

            boxes = []
            
            for pred in result.object_prediction_list:
                try:
                    # Get the class ID of the prediction
                    class_id = int(pred.category.id)
                    
                    # Get the confidence threshold for the current class
                    class_threshold = class_confidence_thresholds.get(class_id, 0.4)  # Default to 0.4 if class not found
                    
                    # Check if the prediction score is below the class-specific threshold
                    if pred.score.value < class_threshold:
                        continue

                    width = int((pred.bbox.maxx - pred.bbox.minx) / scale)
                    height = int((pred.bbox.maxy - pred.bbox.miny) / scale)
                    x = int(pred.bbox.minx / scale)
                    y = int(pred.bbox.miny / scale)

                    bbox_aspect_ratio = (width / height)
                    if bbox_aspect_ratio > bbox_aspect_ratio_threshold or (1 / bbox_aspect_ratio) > bbox_aspect_ratio_threshold:
                        continue 

                    box = [x, y, width, height, class_id]
                    box1 = [x, y, width, height, class_id, pred.score.value]
                    images.append(img_name)
                    box_info.append(box1)
                    area = width * height
                    
                    boxes.append(box)
                    count_1 += 1

                    bbox_start = (int(box[0]), int(box[1]))
                    bbox_end = (int(box[0] + box[2]), int(box[1] + box[3]))

                    draw_bbox(org_img, bbox_start, bbox_end, class_id)
                    
                except Exception as e:
                    print(e)
            cv2.imwrite(os.path.join(dst_path, img_path), org_img)
            print(img_path)

            if boxes:
                image_data[img_name] = {
                    'boxes': boxes
                }

    except Exception as e:
        print(e)
    print("Total Cells Detected: ", count_1)
    df = pd.DataFrame({'image': images,
                        'box_info': box_info})
    df.to_csv(os.path.join(input_folder, "result_sample.csv"), index=False)
    return image_data

# Main execution
start = time.time()
image_data = extract_cells() 

save_detection_counts_to_csv(image_data, output_csv_path)
save_prediction_to_coco_json(image_data, output_json_path)

end = time.time()

print("Total execution time: ", end - start)


# OUTput_Recon_data_130c50aa-b558-4660-ba6e-aee561ab4dfa 
# OUTput_Recon_data_266ae3e8-cd33-48fe-9769-be653e2700f6
# 
