import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np

# pkl_file_path = "/Users/bhavish/rbc_plt_annotations/sigvet_rbc_recon_data/raw_data/0000__013_008.pkl"
# with open(pkl_file_path, 'rb') as f:
#         pkl_data = pickle.load(f)
    
# fov = pkl_data.get('BestFocusedFalseColoredRBCImage')
# img = fov
# #img = cv2.cvtColor(fov, cv2.COLOR_RGB2BGR)

# wl_img = img[:,:,0]
# uv_img = img[:,:,1]

# #plt.imsave("/Users/bhavish/rbc_plt_annotations/sigvet_rbc_recon_data/raw_data/rgb_falsepos_images/image_plt.png", img)
# cv2.imwrite("/Users/bhavish/rbc_plt_annotations/sigvet_rbc_recon_data/raw_data/rgb_falsepos_images/gray_wl_img.png", wl_img)
# cv2.imwrite("/Users/bhavish/rbc_plt_annotations/sigvet_rbc_recon_data/raw_data/rgb_falsepos_images/gray_uv_img.png", uv_img)

def plot_blue_channel_histogram(image):
    blue_channel = image
    hist = cv2.calcHist([blue_channel], [0], None, [256], [0, 256]).flatten()

    # Peak detection using NumPy
    peaks = np.where((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))[0] + 1
    # Apply minimum distance filter between peaks (similar to `distance=10` in find_peaks)
    filtered_peaks = [peaks[0]]
    for peak in peaks[1:]:
        if peak - filtered_peaks[-1] >= 10:
            filtered_peaks.append(peak)
    
    peak_dict = {peak: hist[peak] for peak in filtered_peaks}
    I0 = find_I0(peak_dict)
    
    return I0


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


pkl_file_path = "/Users/bhavish/rbc_plt_annotations/sigvet_rbc_recon_data/raw_data/0000__013_008.pkl"
with open(pkl_file_path, 'rb') as f:
        pkl_data = pickle.load(f)
    
fov = pkl_data.get('BestFocusedFalseColoredRBCImage')
img = fov
plt.imsave("/Users/bhavish/rbc_plt_annotations/sigvet_rbc_recon_data/raw_data/rgb_falsepos_images/fov.png", fov)
#img = cv2.cvtColor(fov, cv2.COLOR_RGB2BGR)

wl_img = img[:,:,0]
uv_img = img[:,:,1]
wl_I0 = plot_blue_channel_histogram(wl_img)
uv_I0 = plot_blue_channel_histogram(uv_img)

ratio = 163.3453871797972/134.33095154088858
bit_depth = 8
wl_img_2 = wl_img * ratio
wl_img_2[wl_img_2>(2**bit_depth-1)] = 2**bit_depth - 1

false_colored_image = np.zeros((np.shape(uv_img)+ (3,)), dtype=np.uint8)
false_colored_image[:,:,0] = wl_img_2
false_colored_image[:,:,1] = uv_img
false_colored_image[:,:,2] = uv_img

#false_colored_image = cv2.cvtColor(false_colored_image, cv2.COLOR_RGB2BGR)
cv2.imwrite("/Users/bhavish/rbc_plt_annotations/sigvet_rbc_recon_data/raw_data/rgb_falsepos_images/false_coloured_img.png", false_colored_image)
cv2.imwrite("/Users/bhavish/rbc_plt_annotations/sigvet_rbc_recon_data/raw_data/rgb_falsepos_images/false_coloured_img.png", false_colored_image)
plt.imsave("/Users/bhavish/rbc_plt_annotations/sigvet_rbc_recon_data/raw_data/rgb_falsepos_images/plt_false_coloured_img.png", false_colored_image)


list_of_intensity_values = uv_img.flatten()
plt.figure(figsize=(8, 6))
plt.hist(list_of_intensity_values, bins=256, range=(0, 255), color='gray', alpha=0.7)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Histogram of Intensity Values")
plt.show()