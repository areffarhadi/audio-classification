import os
import pandas as pd

# Define dataset root and output directory
dataset_root = "/media/rf/T9/voxblink1/arabic_crop2/audio"  # Change this to your actual dataset path
output_dir = "./manifests"
os.makedirs(output_dir, exist_ok=True)

# Collect all files with hierarchical structure
data = []

for root, _, files in os.walk(dataset_root):
    wav_files = [f for f in files if f.endswith(".wav")]
    if wav_files:
        relative_root = os.path.relpath(root, dataset_root)  # Preserve hierarchy
        for f in wav_files:
            file_path = os.path.join(root, f)
            data.append((file_path, relative_root))

# Sort data based on labels
data.sort(key=lambda x: x[1])

# Convert to DataFrame
manifest_df = pd.DataFrame(data, columns=["file_path", "folder_structure"])

# Save manifest as CSV
manifest_df.to_csv(os.path.join(output_dir, "manifest.csv"), index=False, sep=",")

print("Manifest file created successfully in", output_dir)

