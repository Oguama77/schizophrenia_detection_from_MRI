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


'''import os
import shutil
import pandas as pd

# Paths
csv_path = os.path.abspath("selected_files.csv")
source_dir = os.path.abspath("data/processed/extracted_brain")
destination_dir = os.path.abspath("data/processed/extracted_brain/train_set")

# Ensure destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Load the CSV file
df = pd.read_csv(csv_path)

# Move files
counter = 0
for file_name in df["Filename"].str.strip():  # Assuming CSV has a column 'filename'
    file_name = file_name.strip()
    src = os.path.join(source_dir, file_name)
    dst = os.path.join(destination_dir, file_name)

    print(f"Checking file: {repr(file_name)}")  # Debugging step
    print(f"Source path: {src}")
    print(f"Destination path: {dst}")

    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved: {file_name}")
        counter += 1
        print(f"Moved {counter} file(s).")
    else:
        print(f"File not found: {file_name}")

print("All selected files moved successfully!")'''