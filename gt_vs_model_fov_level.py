import json
import pandas as pd
from collections import defaultdict

with open('/Users/bhavish/Downloads/GT_9f9d93ef-685b-4da3-95a1-d9e2a2ff3157-f54e4769-be67-44db-ad6a-3017d2a3e21a__12th__apr/ground_truth_result_real.json', 'r') as f:
    gt_data = json.load(f)

with open('/Users/bhavish/Downloads/GT_9f9d93ef-685b-4da3-95a1-d9e2a2ff3157-f54e4769-be67-44db-ad6a-3017d2a3e21a__12th__apr/yolo-results_rbc_plt_coco.json', 'r') as f:
    pred_data = json.load(f)

image_id_to_name = {img['id']: img['file_name'] for img in gt_data['images']}

gt_counts = defaultdict(lambda: {'rbc': 0, 'wbc': 0, 'plt': 0, 'plt_clump': 0, 'number_of_cells': 0})
for ann in gt_data['annotations']:
    image_id = ann['image_id']
    cat_id = ann['category_id']
    gt_counts[image_id]['number_of_cells'] += 1

    if cat_id == 0:
        gt_counts[image_id]['plt'] += 1
    elif cat_id == 1:
        gt_counts[image_id]['plt_clump'] += 1
    elif cat_id == 2:
        gt_counts[image_id]['rbc'] += 1
    elif cat_id == 3:
        gt_counts[image_id]['wbc'] += 1
    else:
        print(f"Warning: Unknown category_id {cat_id} in ground truth.")

pred_counts = defaultdict(lambda: {'rbc': 0, 'wbc': 0, 'plt': 0, 'plt_clump': 0})

for ann in pred_data['annotations']:
    image_id = ann['image_id']
    cat_id = ann['category_id']
    
    if cat_id == 0:
        pred_counts[image_id]['plt'] += 1
    elif cat_id == 1:
        pred_counts[image_id]['plt_clump'] += 1
    elif cat_id == 2:
        pred_counts[image_id]['rbc'] += 1
    elif cat_id == 3:
        pred_counts[image_id]['wbc'] += 1
    else:
        print(f"Warning: Unknown category_id {cat_id} in prediction.")

results = []

for image_id, file_name in image_id_to_name.items():
    row = {
        'image_name': file_name,
        'number_of_cells_gt': gt_counts[image_id]['number_of_cells'],
        'rbc_gt': gt_counts[image_id]['rbc'],
        'wbc_gt': gt_counts[image_id]['wbc'],
        'plt_gt': gt_counts[image_id]['plt'],
        'plt_clump_gt': gt_counts[image_id]['plt_clump'],
        'rbc_pred': pred_counts[image_id]['rbc'],
        'wbc_pred': pred_counts[image_id]['wbc'],
        'plt_pred': pred_counts[image_id]['plt'],
        'plt_clump_pred': pred_counts[image_id]['plt_clump']
    }
    results.append(row)

df = pd.DataFrame(results)
df.to_csv('comparison_output.csv', index=False)

print("Done! Output saved as 'comparison_output.csv'.")
