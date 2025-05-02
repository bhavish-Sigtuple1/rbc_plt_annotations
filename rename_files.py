import os

# Path to your folder containing the images
folder_path = "/Users/bhavish/Desktop/rbc_plt_iter_7/images"
prefix = "project_7_"  # Prefix to add to each file

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Skip hidden/system files if any
    if filename.startswith('.'):
        continue

    # Create full path
    old_path = os.path.join(folder_path, filename)

    # Ensure it's a file (not a subfolder)
    if os.path.isfile(old_path):
        # Rename to add the prefix
        new_name = prefix + filename
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")
