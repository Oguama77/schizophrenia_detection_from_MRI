import os
import shutil
import pandas as pd

# Paths
csv_file = "selected_files.csv"
src_dir = "data/raw_pt"
dst_dir = "data/raw_pt/basis_for_train_set"

# Ensure destination directory exists
os.makedirs(dst_dir, exist_ok=True)

# Load CSV and extract filenames
df = pd.read_csv(csv_file)
df["Filename"] = df["Filename"].apply(lambda x: os.path.basename(x))  # Extract only filename

# Move files
for filename in df["Filename"]:
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)

    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        print(f"Moved: {filename}")
    else:
        print(f"File not found: {filename}")
