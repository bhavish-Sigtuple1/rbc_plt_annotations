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
import cv2, os, pickle
from glob import glob
import numpy as np

# Model and directories
model_path = "/Users/bhavish/rbc_plt_annotations/rbc_plt_annotations/Models/rbc_plt_iter_12.onnx"

# Class and color mappings
class_mapping = {"0": "plt", "1": "plt-clump", "2": "rbc", "3":"rbc-ghost","4":"rbc-nonspherical" ,"5":"wbc"}

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
    2: 0.6,   # Threshold for class 'rbc'
    3: 0.0, # Threshold for class 'rbc ghost'
    4: 0.0, # Threshold for class 'rbc-nonspherical'
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


scale = 1 / 1
bbox_aspect_ratio_threshold = 1.75

# Function to save detection counts to CSV
def save_detection_counts_to_csv(image_data, output_csv_path):
    # Define the header for the CSV
    header = ['Image_Name', 'plt_count', 'plt_clump_count', 'rbc_count','rbc_ghost_count', 'rbc_nonspherical_count','wbc_count']

    # Open the CSV file in write mode
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
        # Write the counts for each image
        for img_name, data in image_data.items():
            plt_count = sum(1 for box in data['boxes'] if box[4] == 0)  # Count class 0 (plt)
            plt_clump_count = sum(1 for box in data['boxes'] if box[4] == 1)  # Count class 1 (plt-clump)
            rbc_count = sum(1 for box in data['boxes'] if box[4] == 2)  # Count class 2 (rbc)
            rbc_ghost_count = sum(1 for box in data['boxes'] if box[4] == 3)  # Count class 3 (rbc-ghost)
            rbc_nonspherical_count = sum(1 for box in data['boxes'] if box[4] == 4)  # Count class 4 (rbc-nonspherical)
            wbc_count = sum(1 for box in data['boxes'] if box[4] == 5)  # Count class 3 (wbc)
            writer.writerow([img_name,plt_count,plt_clump_count,rbc_count,rbc_ghost_count,rbc_nonspherical_count,wbc_count])

def find_I0(peak_dict):
    weighted_avg_k = 0
    possible_I0 = [i for i in peak_dict.keys()]
    weighted_avg_k = 0
    if len(possible_I0) == 0:
        I0 = 'NA'
    elif len(possible_I0) == 1:
        I0 = possible_I0[0]
    else:
        filtered_dict = {k: v for k, v in peak_dict.items() if k <= 255 and k >= 128}
        if filtered_dict:
            total_weighted_k = sum(k * v for k, v in filtered_dict.items())
            total_occurrences = sum(filtered_dict.values())
            if total_occurrences > 0:
                weighted_avg_k = total_weighted_k / total_occurrences
                                                                                
    return weighted_avg_k
    
def plot_channel_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    # Peak detection using NumPy
    peaks = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0] + 1
    filtered_peaks = [peaks[0]]
    for peak in peaks[1:]:
        if peak - filtered_peaks[-1] >= 10:
            filtered_peaks.append(peak)
    peak_dict = {peak: hist[peak] for peak in filtered_peaks}
    I0 = find_I0(peak_dict)
    return I0
    
def cal_I0_otsu_mapping(img):
    img = np.uint8(img)
    _, binary = cv2.threshold(img, 0, 140, cv2.THRESH_BINARY)
    if np.count_nonzero(binary)==0:
        return 255
    return np.sum(img * (binary / 255)) / np.count_nonzero(binary)


def pkltoimgconvert(img_path, des_dir_extractor):
    all_files = os.listdir(img_path)
    idx = 0
    for pkl_file in all_files:
        if pkl_file.endswith('.pkl'):
            src_file = os.path.join(img_path, pkl_file)
            print(src_file,'src_file')
            with open(src_file, 'rb') as f:
                pkl_data = pickle.load(f)
            
            fov = pkl_data.get('BestFocusedFalseColoredRBCImage')
            wl_img = fov[:,:,0]
            uv_img = fov[:,:,1]
            wl_I0 = cal_I0_otsu_mapping(wl_img)
            uv_I0 = cal_I0_otsu_mapping(uv_img)
            ratio = uv_I0/wl_I0
            wl_img = wl_img * ratio
            wl_img[wl_img>(255)] = 255
            false_colored_image = np.zeros((np.shape(uv_img)+ (3,)), dtype=np.uint8)
            false_colored_image[:,:,0] = wl_img
            false_colored_image[:,:,1] = uv_img
            false_colored_image[:,:,2] = uv_img    

            img = cv2.cvtColor(false_colored_image, cv2.COLOR_RGB2BGR)
            img_name = os.path.basename(src_file)
            
            # cv2.imshow(fov,'img')
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # Determine cropping coordinates (adjust as needed)
            # crop_top, crop_bottom, crop_left, crop_right = 124, 956, 304, 1136
            crop_top, crop_bottom = 0, img.shape[0]    # 0 to 1080
            crop_left, crop_right = 0, img.shape[1]
            
            # Ensure the image is large enough
            # if img.shape[0] < crop_bottom or img.shape[1] < crop_right:
            #     print(f"Skipping {img_name} due to insufficient size.")
            #     continue
            
            # Crop and place the image into a 832x832 canvas
            cropped_image = img[crop_top:crop_bottom, crop_left:crop_right]
            
            # Create a blank image with desired dimensions
            temp_img = np.zeros((1088, 1440, 3), dtype=img.dtype)
            
            temp_img[:cropped_image.shape[0], :cropped_image.shape[1]] = cropped_image
            new_filename = f"{img_name[:-4]}_patch_{idx}.png"
            cv2.imwrite(os.path.join(des_dir_extractor, new_filename), temp_img)
            idx += 1
            print(f"Processed {new_filename}")


# Function to save prediction bbox and class to COCO JSON
def save_prediction_to_coco_json(test_img_dir,image_data,output_json_path):
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
def extract_cells(rbc_path,test_img_dir,dst_path,output_csv_path,output_json_path):
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
    df.to_csv(os.path.join(rbc_path, "result_sample.csv"), index=False)
    return image_data

def main():
    input_folder = "/Users/bhavish/gcp_data"
    main_csv_path = "/Users/bhavish/rbc_plt_annotations/GCP_DATA.csv"
    summary = []
    for folder in os.listdir(input_folder):
        test_img_dir = os.path.join(input_folder, folder)
        if not os.path.isdir(test_img_dir):
            continue
        # find order_id_path safely
        order_id_path = None
        for sub_folder in os.listdir(test_img_dir):
            if sub_folder in ["MacroQCData", "micro_qc_data.pkl", "MicroQCData",".DS_Store"]:
                continue
            print(sub_folder, folder)
            order_id_path = os.path.join(test_img_dir, sub_folder)
            break  # take the first valid subfolder
    #     if not order_id_path or not os.path.exists(order_id_path):
    #         print(f"⚠️ Skipping {folder}, no valid order_id_path found")
    #         continue
    #     recon_data_path = os.path.join(order_id_path, "rbc", "Recon_data")
    #     rbc_path = os.path.join(order_id_path, "rbc")
    #     destination_folder_path = os.path.join(rbc_path, "output_1088x1440")
    #     os.makedirs(destination_folder_path, exist_ok=True)
    #     # ensure Recon_data exists
    #     if os.path.exists(recon_data_path):
    #         pkltoimgconvert(recon_data_path, destination_folder_path)
    #     else:
    #         print(f"⚠️ No Recon_data for {folder}, skipping conversion")

    #     dst_path = os.path.join(rbc_path, "yolo-results")
    #     os.makedirs(dst_path, exist_ok=True)
    #     output_csv_path = os.path.join(rbc_path, "yolo-results.csv")
    #     output_json_path = os.path.join(rbc_path, "yolo-results_rbc_plt_coco.json")
    #     image_data = extract_cells(rbc_path, destination_folder_path, dst_path, output_csv_path, output_json_path)
    #     # process CSV only if it exists
    #     save_detection_counts_to_csv(image_data, output_csv_path)
    #     save_prediction_to_coco_json(destination_folder_path, image_data, output_json_path)
    #     if not os.path.exists(output_csv_path):
    #         print(f"⚠️ No output CSV for {folder}, skipping")
    #         continue
    #     data_frame = pd.read_csv(output_csv_path)
    #     plt_total_count = data_frame["plt_count"].sum()
    #     rbc_total_count = data_frame["rbc_count"].sum()
    #     rbc_nonspherical_count = data_frame["rbc_nonspherical_count"].sum()
    #     rbc_ghost_count = data_frame["rbc_ghost_count"].sum()
    #     # save results
    #     # add summary entry
    #     summary.append({
    #         "order_id": folder,
    #         "plt_total_count": plt_total_count,
    #         "rbc_total_count": rbc_total_count,
    #         "rbc_nonspherical_count": rbc_nonspherical_count,
    #         "rbc_ghost_count": rbc_ghost_count
    #     })
    # # save master summary
    # summary_df = pd.DataFrame(summary)
    # if not os.path.exists(main_csv_path):
    #     summary_df.to_csv(main_csv_path, index=False)
    # else:
    #     summary_df.to_csv(main_csv_path, mode="a", header=False, index=False)

if __name__ == "__main__":
    main()
