import pandas as pd

# Update this path with the actual CSV file location
input_csv_path = "predictions_mp4.csv"

# Load the CSV file
df = pd.read_csv(input_csv_path)

# Filter rows where "predicted_speaker" is not "Speech"
df_not_speech = df[df["predicted_speaker"] != "Speech"]

# Filter rows where "predicted_speaker" is "Speech" but "confidence" is not more than 0.5
df_low_confidence_speech = df[(df["predicted_speaker"] == "Speech") & (df["confidence"] <= 0.5)]

# Define output file paths
output_not_speech = "not_speech.csv"
output_low_confidence_speech = "low_confidence_speech.csv"

# Save the filtered data to new CSV files
df_not_speech.to_csv(output_not_speech, index=False)
df_low_confidence_speech.to_csv(output_low_confidence_speech, index=False)

print(f"Filtered files saved:\n- {output_not_speech}\n- {output_low_confidence_speech}")

