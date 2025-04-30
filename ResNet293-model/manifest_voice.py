import os
import pandas as pd

def generate_manifest(base_directory, output_csv):
    rows = []
    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                # Extract speaker ID from the folder name two levels up
                speaker_id = os.path.basename(os.path.dirname(file_path))
      
                rows.append({'wav_path': file_path, 'speaker_id': speaker_id})
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    base_directory = "./test_spkr10/train"
    output_csv = "./manifest_train.csv"
    generate_manifest(base_directory, output_csv)

