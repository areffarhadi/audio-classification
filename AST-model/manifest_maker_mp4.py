import os
import pandas as pd

# Define dataset root and output directory
dataset_root = "/home/rf/voxblink2/ScriptsForVoxBlink2/urdu_speakers_clip"  # Change this to your actual dataset path
output_dir = "./manifests_mp4"
os.makedirs(output_dir, exist_ok=True)

# Collect all MP4 files with hierarchical structure
data = []
for root, _, files in os.walk(dataset_root):
    mp4_files = [f for f in files if f.lower().endswith(".mp4")]
    if mp4_files:
        relative_root = os.path.relpath(root, dataset_root)  # Preserve hierarchy
        for f in mp4_files:
            file_path = os.path.join(root, f)
            data.append((file_path, relative_root))

# Sort data based on labels
data.sort(key=lambda x: x[1])

# Convert to DataFrame
manifest_df = pd.DataFrame(data, columns=["file_path", "folder_structure"])

# Save manifest as CSV
manifest_path = os.path.join(output_dir, "mp4_manifest.csv")
manifest_df.to_csv(manifest_path, index=False, sep=",")
print(f"Manifest file created successfully at: {manifest_path}")
print(f"Found {len(data)} MP4 files in {dataset_root}")
