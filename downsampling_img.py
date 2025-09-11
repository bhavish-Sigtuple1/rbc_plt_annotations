
#### Downsampling the image

import os 
import cv2 

file_path = "/Users/bhavish/Documents/rbc_plt_iter_10/Data_for_Annotation/High_tc_samples/4f41f709-90dc-47b6-ac11-122ce07b595c/output_1080_1440"

des_path = "/Users/bhavish/Documents/rbc_plt_iter_10/Data_for_Annotation/High_tc_samples/4f41f709-90dc-47b6-ac11-122ce07b595c/downsampled_train2017"

os.makedirs(des_path, exist_ok=True)

for x in os.listdir(file_path):
    if x.endswith(".png"):
        img = cv2.imread(os.path.join(file_path, x))
        height, width = img.shape[:2]
        img = cv2.resize(img, (int(width * (0.5)), int(height * (0.5))))
        cv2.imwrite(os.path.join(des_path, x), img)

# import json

# with open("/Users/bhavish/Downloads/rbc_plt_iter_9_1_training_data/annotations/train2017.json", "r") as f:
#     data = json.load(f)

# for ann in data["annotations"]:
#     ann["bbox"] = [x / 2 for x in ann["bbox"]]
#     ann["area"] = ann["area"] / 4

# # save back
# with open("/Users/bhavish/Downloads/rbc_plt_iter_9_1_training_data/annotations/train_updated.json", "w") as f:
#     json.dump(data, f, indent=4)


# Padding the image
# import cv2
# import os
# import glob

# # input and output folders
# input_dir = "/Users/bhavish/Downloads/rbc_plt_iter_9_1_training_data/rbc_plt_iter_9_1_1x_training/train2017"
# output_dir = "/Users/bhavish/Downloads/rbc_plt_iter_9_1_training_data/rbc_plt_iter_9_1_1x_training/train2017_padding"
# os.makedirs(output_dir, exist_ok=True)

# # loop through all images
# for file in glob.glob(os.path.join(input_dir, "*.png")):  # change to jpg if needed
#     img = cv2.imread(file)
#     h, w = img.shape[:2]

#     # calculate padding values
#     top = (224 - h) // 2
#     bottom = 224 - h - top
#     left = (224 - w) // 2
#     right = 224 - w - left

#     # pad image with black borders
#     padded = cv2.copyMakeBorder(img, top, bottom, left, right,
#                                 cv2.BORDER_CONSTANT, value=[0, 0, 0])

#     # save padded image
#     filename = os.path.basename(file)
#     cv2.imwrite(os.path.join(output_dir, filename), padded)

# print("✅ Padding done. All images are now 224×224 and saved to:", output_dir)


# import json

# # input/output JSON
# input_json = "/Users/bhavish/Downloads/rbc_plt_iter_9_1_training_data/rbc_plt_iter_9_1_1x_training/annotations/train2017.json"
# output_json = "/Users/bhavish/Downloads/rbc_plt_iter_9_1_training_data/rbc_plt_iter_9_1_1x_training/annotations/train2017_updated.json"

# # fixed padding values
# left_pad = 8
# top_pad = 8

# with open(input_json, "r") as f:
#     data = json.load(f)

# # update images size
# for img in data["images"]:
#     img["width"] = 224
#     img["height"] = 224

# # shift bboxes
# for ann in data["annotations"]:
#     ann["bbox"][0] += left_pad  # shift x
#     ann["bbox"][1] += top_pad   # shift y
#     # width and height (bbox[2], bbox[3]) stay the same

# # save back
# with open(output_json, "w") as f:
#     json.dump(data, f, indent=4)

# print("✅ JSON updated: image size → 224×224, bbox x/y shifted by 8px")
