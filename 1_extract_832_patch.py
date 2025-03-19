import cv2, os, pickle
from glob import glob
import numpy as np

img_path = '/Users/bhavish/rbc_plt_annotations/Recon_data_2'
des_dir_extractor = "/Users/bhavish/rbc_plt_annotations/des_dir_path"
os.makedirs(des_dir_extractor, exist_ok=True)

all_files = os.listdir(img_path)

idx = 0
for pkl_file in all_files:
    src_file = os.path.join(img_path, pkl_file)
    with open(src_file, 'rb') as f:
        pkl_data = pickle.load(f)
    
    fov = pkl_data.get('BestFocusedFalseColoredRBCImage')
    img = cv2.cvtColor(fov, cv2.COLOR_RGB2BGR)
    img_name = os.path.basename(src_file)
    
    # cv2.imshow(fov,'img')
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Determine cropping coordinates (adjust as needed)
    crop_top, crop_bottom, crop_left, crop_right = 124, 956, 304, 1136
    
    # Ensure the image is large enough
    if img.shape[0] < crop_bottom or img.shape[1] < crop_right:
        print(f"Skipping {img_name} due to insufficient size.")
        continue
    
    # Crop and place the image into a 832x832 canvas
    cropped_image = img[crop_top:crop_bottom, crop_left:crop_right]
    
    # Create a blank image with desired dimensions
    temp_img = np.zeros((832, 832, 3), dtype=img.dtype)
    
    # Place the cropped image into the blank canvas
    temp_img[:cropped_image.shape[0], :cropped_image.shape[1]] = cropped_image
    
    # Save the result with a new name
    new_filename = f"{img_name[:-4]}_patch_{idx}.png"
    cv2.imwrite(os.path.join(des_dir_extractor, new_filename), temp_img)
    
    idx += 1
    print(f"Processed {new_filename}")
