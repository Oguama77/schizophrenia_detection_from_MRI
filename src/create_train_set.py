import os
import shutil
import pandas as pd

# Get absolute paths
csv_path = os.path.abspath("selected_files.csv")
destination_dir = os.path.abspath("data/processed/extracted_brain/train_set")

# Ensure destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Load the CSV file
df = pd.read_csv(csv_path)

# Move files
for file_path in df["Filename"]:  # Adjust column name if needed
    file_path = file_path.strip()  # Remove any unwanted spaces

    # Convert relative path from CSV to absolute path
    src = os.path.abspath(file_path)
    dst = os.path.join(destination_dir, os.path.basename(file_path))

    print(f"Checking file: {repr(file_path)}")  # Debugging step
    print(f"Source path: {src}")
    print(f"Destination path: {dst}")

    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved: {file_path}")
    else:
        print(f"File not found: {src}")

print("All selected files processed!")
