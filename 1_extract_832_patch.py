import cv2, os, pickle
from glob import glob
import numpy as np

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


img_path = '/Users/bhavish/Documents/Plt_clump_and_plt_dual_plt/Plt_clumps_data'
des_dir_extractor = "/Users/bhavish/Documents/Plt_clump_and_plt_dual_plt/output_832"
os.makedirs(des_dir_extractor, exist_ok=True)

all_files = os.listdir(img_path)

idx = 0
for pkl_file in all_files:
    if pkl_file == ".DS_Store":
        continue
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
