import json

def coco_to_label_studio_format(coco_json_path, output_json_path):
    with open(coco_json_path, 'r') as f:
        coco_json = json.load(f)

    label_studio_data = [] 
    category_mapping = {1: "plt",2: "plt-clump",3: "rbc",4: "rbc-ghost",5:"rbc-nonspherical",6: "wbc"}
    # category_mapping = {1: "plt",2: "plt-clump",3: "rbc",4: "rbc-ghost",5:"rbc-nonspherical",6: "wbc"}
    for image_info in coco_json['images']:
        image_id = image_info['id']
        annotations_for_image = [ann for ann in coco_json['annotations'] if ann['image_id'] == image_id]
        
        if annotations_for_image:
            image_path = "/data/local-files/?d=data/" + image_info['file_name']
            results = []
            for annotation in annotations_for_image:
                annotation_data = {
                    "id": str(annotation['id']),  # ID should be a string
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": 416,  # Use actual width if known
                    "original_height": 416,  # Use actual height if known
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,
                        "x": (((annotation['bbox'][0]) / 1440) * 100)+0,  # Convert pixel to percentage
                        "y": (((annotation['bbox'][1]) / 1088) * 100)+0,  # Convert pixel to percentage
                        "width": ((annotation['bbox'][2]) / 1440) * 100,  # Convert pixel to percentage
                        "height": ((annotation['bbox'][3]) / 1088) * 100,  # Convert pixel to percentage
                        # "x": (((annotation['bbox'][0]) / 416) * 100)+0,  # Convert pixel to percentage
                        # "y": (((annotation['bbox'][1]) / 416) * 100)+0,  # Convert pixel to percentage
                        # "width": ((annotation['bbox'][2]) / 416) * 100,  # Convert pixel to percentage
                        # "height": ((annotation['bbox'][3]) / 416) * 100,  # Convert pixel to percentage
                        # "x": ((annotation['bbox'][0]) / 416) * 100,  # Convert pixel to percentage
                        # "y": ((annotation['bbox'][1]) / 416) * 100,  # Convert pixel to percentage
                        # "width": ((annotation['bbox'][2]) / 416) * 100,  # Convert pixel to percentage
                        # "height": ((annotation['bbox'][3]) / 416) * 100,  # Convert pixel to percentage
                        #"rectanglelabels": ["rbc"]
                        "rectanglelabels": [category_mapping[annotation['category_id']]]
                    }
                }
                results.append(annotation_data)
            
            label_studio_annotation = {
                "data": {
                    "image": image_path
                },
                "predictions": [
                    {
                        "model_version": "rbc_expt",
                        "score": 0.5,
                        "result": results
                    }
                ]
            }
            label_studio_data.append(label_studio_annotation)

    with open(output_json_path, 'w') as f:
        json.dump(label_studio_data, f, indent=4)
    
    # return label_studio_data

coco_annoation_path = "/Users/bhavish/Downloads/Gt_Rbc_PLT_iter_13_2-at-2025-11-03-08-41-3d1dfc74/yolo-results_rbc_plt_coco-iter_13_2.json"
coco_to_label_studio_format(coco_annoation_path, "/Users/bhavish/Downloads/Gt_Rbc_PLT_iter_13_2-at-2025-11-03-08-41-3d1dfc74/GT_json_13_2.json")

