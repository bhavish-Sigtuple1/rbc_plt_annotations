from pycocotools.coco import COCO
import json,os


def merge_coco_json(json_paths, output_path):
    # Initialize merged COCO dataset
    merged_coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_id_map = {}  # Map to store mapping from old image_id to new image_id

    for idx, json_path in enumerate(json_paths):
        coco = COCO(json_path)

        # Update info and licenses if missing in merged_coco
        if idx == 0:
            merged_coco["info"] = coco.dataset.get("info", {})
            merged_coco["licenses"] = coco.dataset.get("licenses", [])

        # Add images from the current dataset
        for img in coco.dataset["images"]:
            new_img_id = len(merged_coco["images"]) + 1
            image_id_map[(idx, img["id"])] = new_img_id

            # Extract filename from file_name (assuming it's a full path)
            img_filename = os.path.basename(img["file_name"])
            img["file_name"] = img_filename

            img["id"] = new_img_id
            merged_coco["images"].append(img)

        # Add annotations from the current dataset
        for ann in coco.dataset["annotations"]:
            new_ann = ann.copy()
            new_ann["image_id"] = image_id_map[(idx, ann["image_id"])]
            merged_coco["annotations"].append(new_ann)

        # Update categories (assuming they are the same across all datasets)
        if idx == 0:
            merged_coco["categories"] = coco.dataset["categories"]

    # Create a new COCO object
    merged_coco_obj = COCO()
    merged_coco_obj.dataset = merged_coco
    merged_coco_obj.createIndex()

    # Save merged dataset to JSON file
    with open(output_path, "w") as f:
        json.dump(merged_coco_obj.dataset, f)

    print(f"Merged COCO JSON saved to {output_path}")



json_paths = ['/Users/bhavish/rbc_plt_annotations/result.json',
              '/Users/bhavish/Desktop/Rbc_plt_iter_6/rbc_plt_iter_5/val2017.json'
              ]
output_path = "/Users/bhavish/Desktop/Rbc_plt_iter_6/train2017.json"

merge_coco_json(json_paths, output_path)
