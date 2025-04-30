import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from src.models import ASTModel
import torch.nn as nn
from tqdm import tqdm

# Hardcoded configuration
MODEL_PATH = "ast_finetuned.pth"  # Path to your fine-tuned model
TEST_MANIFEST = "/home/rf/AudioSet/ast/manifests/manifest.csv"  # Path to your test manifest file
OUTPUT_CSV = "predictions.csv"  # Output file for results
NUM_CLASSES = 3  # Number of speaker classes

# Hardcoded class mapping
CLASS_MAPPING = {0: "Music", 1: "Non-Speech", 2: "Speech"}  # Replace with your actual classes

class FineTunedAST(nn.Module):
    def __init__(self, pretrained_model_path, num_classes):
        super(FineTunedAST, self).__init__()
        
        # Load pretrained AST model
        self.ast_model = ASTModel(label_dim=527, input_tdim=1024, 
                                 imagenet_pretrain=False, audioset_pretrain=False)
        self.ast_model = nn.DataParallel(self.ast_model)
        
        # Replace the classifier with a new one for our target classes
        embedding_dim = self.ast_model.module.original_embedding_dim
        
        # Create new classifier
        self.new_classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )
        
    def forward(self, x):
        # Forward through the AST model but stop before the classifier
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.ast_model.module.v.patch_embed(x)
        cls_tokens = self.ast_model.module.v.cls_token.expand(B, -1, -1)
        dist_token = self.ast_model.module.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.ast_model.module.v.pos_embed
        x = self.ast_model.module.v.pos_drop(x)
        
        for blk in self.ast_model.module.v.blocks:
            x = blk(x)
            
        x = self.ast_model.module.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2  # Average of cls and dist tokens
        
        # Use our new classifier
        x = self.new_classifier(x)
        return x

def extract_features(wav_path, mel_bins=128, target_length=1024):
    try:
        waveform, sr = torchaudio.load(wav_path)
        
        # Ensure 16kHz sampling rate
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
            
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Compute Mel Spectrogram
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=16000, use_energy=False,
            window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10
        )
        
        # Pad or trim to target length
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
        elif p < 0:
            fbank = fbank[:target_length, :]
            
        # Normalize features - using the same values as in the training script
        fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
        
        return fbank
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None

def single_file_inference(audio_file_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get class names from the mapping
    classes = [CLASS_MAPPING[i] for i in range(len(CLASS_MAPPING))]
    print(f"Class mapping: {CLASS_MAPPING}")
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = FineTunedAST(None, NUM_CLASSES)  # We'll load weights after
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # Process audio file
    print(f"Processing audio file: {audio_file_path}")
    features = extract_features(audio_file_path)
    
    if features is None:
        print("Failed to extract features from the audio file.")
        return
    
    # Make prediction
    with torch.no_grad():
        features = features.unsqueeze(0).to(device)  # Add batch dimension
        outputs = model(features)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probabilities, 1)
        
        predicted_class = CLASS_MAPPING[prediction.item()]
        confidence_value = confidence.item() * 100
        
        print(f"\nPrediction Results:")
        print(f"Predicted speaker: {predicted_class}")
        print(f"Confidence: {confidence_value:.2f}%")
        
        # Show all class probabilities
        print("\nProbabilities for all classes:")
        probs = probabilities.squeeze().cpu().numpy()
        for i, prob in enumerate(probs):
            print(f"{CLASS_MAPPING[i]}: {prob*100:.2f}%")

def manifest_inference():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the test manifest
    try:
        test_manifest = pd.read_csv(TEST_MANIFEST)
        print(f"Loaded test manifest with {len(test_manifest)} entries")
    except Exception as e:
        print(f"Error loading test manifest: {e}")
        return
    
    # Get class names from the mapping
    classes = [CLASS_MAPPING[i] for i in range(len(CLASS_MAPPING))]
    print(f"Class mapping: {CLASS_MAPPING}")
    
    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = FineTunedAST(None, NUM_CLASSES)  # We'll load weights after
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # Process all files in the manifest
    results = []
    
    for _, row in tqdm(test_manifest.iterrows(), total=len(test_manifest), desc="Processing files"):
        file_path = row['file_path']
        true_speaker = row.get('speaker_id', 'unknown')  # Get ground truth if available
        
        features = extract_features(file_path)
        
        if features is None:
            print(f"Skipping {file_path} due to processing error")
            continue
        
        # Make prediction
        with torch.no_grad():
            features = features.unsqueeze(0).to(device)  # Add batch dimension
            outputs = model(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
            predicted_class_idx = prediction.item()
            predicted_class = CLASS_MAPPING[predicted_class_idx]
            confidence_value = confidence.item()
            
            # Store all class probabilities
            probs = {f"prob_{CLASS_MAPPING[i]}": prob.item() for i, prob in enumerate(probabilities.squeeze())}
            
            result = {
                'file_path': file_path,
                'true_speaker': true_speaker,
                'predicted_speaker': predicted_class,
                'predicted_class_idx': predicted_class_idx,
                'confidence': confidence_value,
                **probs
            }
            
            results.append(result)
    
    # Create and save dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")
    
    # Calculate accuracy if ground truth is available
    if 'true_speaker' in results_df.columns and results_df['true_speaker'].nunique() > 1:
        correct = 0
        total = 0
        for _, row in results_df.iterrows():
            if row['true_speaker'] != 'unknown':
                total += 1
                if row['true_speaker'] == row['predicted_speaker']:
                    correct += 1
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"\nAccuracy on test set: {accuracy:.2f}% ({correct}/{total})")

def main():
    print("\nAST Speaker Identification Inference")
    print("=====================================")
    print("1. Run inference on a single audio file")
    print("2. Run inference on all files in test manifest")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        audio_file = input("Enter the path to the audio file: ")
        single_file_inference(audio_file)
    elif choice == '2':
        manifest_inference()
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
