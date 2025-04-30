import pandas as pd
import os

# Update this path with the actual CSV file location
input_csv_path = "predictions.csv"

# Load the CSV file
df = pd.read_csv(input_csv_path)

# Filter rows where "predicted_speaker" is not "Speech"
df_not_speech = df[df["predicted_speaker"] != "Speech"]

# Iterate through the rows and delete files
for index, row in df_not_speech.iterrows():
    wav_path = row["file_path"]
    mp4_path = wav_path.replace("/audio/", "/video/").replace(".wav", ".mp4")

    # Remove the .wav file
    if os.path.exists(wav_path):
        os.remove(wav_path)
        print(f"Deleted: {wav_path}")
    else:
        print(f"File not found: {wav_path}")

    # Remove the corresponding .mp4 file
    if os.path.exists(mp4_path):
        os.remove(mp4_path)
        print(f"Deleted: {mp4_path}")
    else:
        print(f"File not found: {mp4_path}")

print("File removal process completed.")

