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
mount_root = "/Volumes/Public"
target_subdir = "Sigvet_DC/PD8"
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
order_ids = ["d51d5cac-8859-4221-af61-a3a3cc115f5d","12eac653-ec9a-4451-ab32-aafc7dfcf290"]


data_path = mount_point
destination_base = "/Users/bhavish/Downloads/destination_base" # here the destination from the NAS


# Traverse NAS directories safely
for accession_number in os.listdir(data_path):
    accession_path = os.path.join(data_path, accession_number)
    if not os.path.isdir(accession_path):
        continue  # skip non-directories like .DS_Store

    for sample_id in os.listdir(accession_path):
        sample_id_path = os.path.join(accession_path, sample_id)
        if not os.path.isdir(sample_id_path):
            continue

        for date in os.listdir(sample_id_path):
            date_path = os.path.join(sample_id_path, date)
            if not os.path.isdir(date_path):
                continue

            for tray_id in os.listdir(date_path):
                tray_id_path = os.path.join(date_path, tray_id)
                if not os.path.isdir(tray_id_path):
                    continue

                for order_id in os.listdir(tray_id_path):
                    order_id_path = os.path.join(tray_id_path, order_id)
                    if not os.path.isdir(order_id_path):  
                        continue  # skip files inside tray_id

                    if order_id not in order_ids:
                        continue

                    recon_path = os.path.join(order_id_path, "rbc", "Recon_data")
                    if not os.path.exists(recon_path):
                        print(f"Recon_data not found for {order_id}")
                        continue

                    dest_folder = os.path.join(destination_base, order_id,"Recon_data")
                    os.makedirs(dest_folder, exist_ok=True)

                    for filename in os.listdir(recon_path):
                        if filename.endswith(".pkl"):
                            src_file = os.path.join(recon_path, filename)
                            dst_file = os.path.join(dest_folder, filename)
                            if os.path.isfile(src_file):
                                shutil.copy(src_file, dst_file)
                        else:
                            continue
