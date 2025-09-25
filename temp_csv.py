import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def plot_hist_with_sd(data, title, color="blue",save_path="None"):
    if not data:
        print(f"No data to plot for {title}")
        return
    
    # Convert to numpy for stats
    arr = np.array(data)
    mean = np.mean(arr)
    std = np.std(arr)

    plt.figure(figsize=(8, 5))
    plt.hist(arr, bins=15, color=color, alpha=0.7, edgecolor="black")

    # Mean line
    plt.axvline(mean, color="red", linestyle="-", linewidth=2, label=f"Mean = {mean:.2f}")

    # ±1 SD lines (dotted)
    plt.axvline(mean - std, color="green", linestyle="--", linewidth=1.5, label=f"-1 SD = {mean-std:.2f}")
    plt.axvline(mean + std, color="green", linestyle="--", linewidth=1.5, label=f"+1 SD = {mean+std:.2f}")

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # make dir if not exists
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {save_path}")
    else:
        plt.show()  

def mean_within_1sd_right(data):
        if not data:
            return 0  
        arr = np.array(data)
        mean = np.mean(arr) # this is the no of plts you get in one fov on avg
        std = np.std(arr) 
        filtered = arr[arr <= (mean + 1.5*std)] 
        # new_mean = np.mean(filtered) if len(filtered) > 0 else 0 # this is no of plts you get in one fov from range [starting, mean + 1sd]
        new_mean = mean + 2*std
        return mean, std, new_mean, filtered


def calculate_tc_n_volumes(csv_folder):
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    plt_clump = 0

    rbc_volumes = []
    plt_volumes = []

    rbc_tc_raw = []
    plt_tc_raw = []

    # First pass: collect counts and volumes
    for file in csv_files:
        file_path = os.path.join(csv_folder, file)
        df = pd.read_csv(file_path)

        # RBC
        df_rbc = df[df['class'] == 2] # This will list all the no of rbc's in fov
        rbc_tc_raw.append(len(df_rbc))
        rbc_volumes.extend(df_rbc.loc[df_rbc['volume'] != 0, 'volume'].tolist())

        # Platelets
        df_plt = df[df['class'] == 0]
        plt_tc_raw.append(len(df_plt))
        plt_volumes.extend(df_plt['volume'].tolist())

        # Platelet clumps
        df_plt_clump = df[df['class'] == 1]
        plt_clump += len(df_plt_clump)

    plot_hist_with_sd(rbc_tc_raw, "RBC Counts per FOV", color="blue",save_path="/Users/bhavish/rbc_plt_annotations/rbc_tc.png")
    plot_hist_with_sd(plt_tc_raw, "Platelet Counts per FOV", color="orange",save_path="/Users/bhavish/rbc_plt_annotations/plt_tc.png")

    mean, std, new_mean, filtered = mean_within_1sd_right(plt_tc_raw)
    print(mean,new_mean)

    image_name = []

    for file in csv_files:
        file_path = os.path.join(csv_folder, file)
        df = pd.read_csv(file_path)

        # Platelets
        df_plt = df[df['class'] == 0]
        if len(df_plt) > new_mean:
            image_name.extend(df_plt['image_name'].tolist())
    image_name = list(set(image_name))
    print(image_name)

    
csv_folder = "/Users/bhavish/Desktop/coordinates_csv_files"
calculate_tc_n_volumes(csv_folder)


