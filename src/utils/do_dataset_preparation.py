import os
import random
import shutil
import pandas as pd

def prepare_dataset(input_dir, train_dir, num_train_images):
    os.makedirs(train_dir, exist_ok=True)

    all_files = [f for f in os.listdir(input_dir) if f.endswith(".pt")]
    selected_files = random.sample(all_files, min(num_train_images, len(all_files)))

    # Save selected filenames
    df = pd.DataFrame(selected_files, columns=["Filename"])
    df.to_csv("selected_files.csv", index=False)

    for file in selected_files:
        src = os.path.join(input_dir, file)
        dst = os.path.join(train_dir, file)
        shutil.copy(src, dst)
        print(f"Copied {file} to train set")
