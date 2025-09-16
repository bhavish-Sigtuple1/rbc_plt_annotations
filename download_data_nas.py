import subprocess
import shutil
import urllib.parse
import pandas as pd
import os
import shutil

# Config
nas_ip = "192.168.0.253"
nas_user = "Sigtuple"
nas_password = "Sigtuple@123"
nas_password_encoded = urllib.parse.quote(nas_password)  # URL encode special characters
nas_share = "Micro-POC"
mount_root = "/Volumes/Micro-POC"
target_subdir = "Sigvet_MicroPOC/SigVet_DS/HC_Devices_data/Mcv_data"
mount_point = os.path.join(mount_root, target_subdir)

def is_mounted(): 
    return os.path.ismount(mount_root)

def mount_nas():
    if is_mounted():
        print(f"{mount_root} is already mounted.")
        return
    print(f"Mounting NAS share '{nas_share}' to {mount_root}...")
    try:
        script = f'''
        osascript -e 'mount volume "smb://{nas_user}:{nas_password_encoded}@{nas_ip}/{nas_share}"'
        '''
        subprocess.run(script, shell=True, check=True)
        print("Mounted successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to mount NAS:", e)


mount_nas()

# Load order IDs
data = pd.read_excel('/imgarc/sigvet/Mcv_calculation/DATA_MCV/Combined_excels/HC3_output.xlsx')
print(data.columns)
csv_order_ids = []
for x in data['Order ID'].tolist():
    if str(x).strip() == 'nan':
        continue
    csv_order_ids.append(str(x).strip())

data_path = mount_point
destination_base = "/imgarc/sigvet/Mcv_calculation/DATA_MCV_HC3" # here the destination from the NAS



for i, folder in enumerate(os.listdir(data_path)):
    if len(os.listdir(os.path.join(data_path,folder)))>7:
           print(folder)
    print(i, folder)
    if folder == "0fd596ca-09d8-495b-8e5b-2518cc1713f4":
        continue
    folder_path = os.path.join(data_path, folder)
    if not os.path.isdir(folder_path):
        print(f"Skipping {folder_path} as it is not a directory.")
        continue

    print(f"Processing folder: {folder_path}")
    for subfolder in os.listdir(folder_path):
        if str(subfolder).strip() in csv_order_ids:
            print(f"Processing order: {subfolder}")
            subfolder_path = os.path.join(folder_path, subfolder)
            
            rbc_path = os.path.join(subfolder_path, "rbc")
            if not os.path.isdir(rbc_path):
                print(f"Skipping {rbc_path} as it does not exist.")
                continue

            recon_path = os.path.join(rbc_path, "Recon_data")
            if not os.path.isdir(recon_path):
                print(f"Skipping {recon_path} as it does not exist.")
                continue

            dest_folder = os.path.join(destination_base, subfolder)
            os.makedirs(dest_folder, exist_ok=True)

            for filename in os.listdir(recon_path):
                src_file = os.path.join(recon_path, filename)
                dst_file = os.path.join(dest_folder, filename)
                if os.path.isfile(src_file):
                    # print(f"Copying {src_file} to {dst_file}")
                    shutil.copy(src_file, dst_file)
                else:
                    print(f"Skipping non-file {src_file}")
        else:
            print(f"Order {subfolder} not in order_id list, skipping.")
            print(folder_path+"/"+subfolder)