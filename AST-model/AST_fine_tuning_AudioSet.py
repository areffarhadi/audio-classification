import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.models import ASTModel

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
TRAINING_STRATEGY = "two_step" 
# Options:
#  - "two_step"       (Stage 1: freeze backbone & train classifier; Stage 2: unfreeze & fine-tune all)
#  - "classifier_only" (Train only the classifier layer for all epochs)

PRETRAINED_MODEL_PATH = "/local/scratch/arfarh/AV-Voxblink-finetune-main/audioset_10_10_0.4593.pth"
TRAIN_MANIFEST = "/local/scratch/arfarh/AV-Voxblink-finetune-main/manifests_audioSet/train_manifest.csv"
VAL_MANIFEST = "/local/scratch/arfarh/AV-Voxblink-finetune-main/manifests_audioSet/val_manifest.csv" 
TEST_MANIFEST = "/local/scratch/arfarh/AV-Voxblink-finetune-main/manifests_audioSet/test_manifest.csv"
NUM_CLASSES = 3
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
NUM_EPOCHS = 10
SAVE_MODEL_PATH = "ast_finetuned.pth"
# -----------------------------------------------------------------

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Training strategy: {TRAINING_STRATEGY}")

# Custom Dataset
class AudioDataset(Dataset):
    def __init__(self, manifest_path, mel_bins=128, target_length=1024):
        self.manifest = pd.read_csv(manifest_path)
        self.mel_bins = mel_bins
        self.target_length = target_length
        # Get unique classes and create label mapping
        self.classes = sorted(self.manifest['speaker_id'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        print(f"Class mapping: {self.class_to_idx}")
        
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        file_path = self.manifest.iloc[idx]['file_path']
        label = self.manifest.iloc[idx]['speaker_id']
        label_idx = self.class_to_idx[label]
        
        # Extract features
        features = self.extract_features(file_path)
        
        return features, label_idx
    
    def extract_features(self, wav_path):
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
                window_type='hanning', num_mel_bins=self.mel_bins, dither=0.0, frame_shift=10
            )
            
            # Pad or trim to target length
            n_frames = fbank.shape[0]
            p = self.target_length - n_frames
            if p > 0:
                fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
            elif p < 0:
                fbank = fbank[:self.target_length, :]
                
            # Normalize features
            fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
            
            return fbank
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            # Return zeros as a fallback
            return torch.zeros(self.target_length, self.mel_bins)

# Fine-tuned AST Model for 3 classes
class FineTunedAST(nn.Module):
    def __init__(self, pretrained_model_path, num_classes):
        super(FineTunedAST, self).__init__()
        
        # Load pretrained AST model
        print("Loading pretrained AST model...")
        self.ast_model = ASTModel(label_dim=527, input_tdim=1024, 
                                  imagenet_pretrain=False, audioset_pretrain=False)
        self.ast_model = nn.DataParallel(self.ast_model)
        self.ast_model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'), strict=False)
        
        # Replace the classifier (mlp_head) with a new one for our target classes
        embedding_dim = self.ast_model.module.original_embedding_dim
        print(f"AST embedding dimension: {embedding_dim}")
        
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

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in tqdm(train_loader, desc="Training"):
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(train_loader), 100. * correct / total

# Validation/Testing function
def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(data_loader, desc="Evaluating"):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'loss': running_loss / len(data_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    # Create the model
    model = FineTunedAST(PRETRAINED_MODEL_PATH, NUM_CLASSES)
    model.to(device)
    
    # Create datasets and dataloaders
    print("Loading datasets...")
    train_dataset = AudioDataset(TRAIN_MANIFEST)
    val_dataset = AudioDataset(VAL_MANIFEST)
    test_dataset = AudioDataset(TEST_MANIFEST)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Print class distribution in training set
    print("Class distribution in training set:")
    train_labels = [train_dataset.class_to_idx[label] for label in train_dataset.manifest['speaker_id']]
    for class_name, class_idx in train_dataset.class_to_idx.items():
        count = train_labels.count(class_idx)
        print(f"  {class_name}: {count} samples ({count/len(train_labels)*100:.2f}%)")
    
    criterion = nn.CrossEntropyLoss()
    
    if TRAINING_STRATEGY == "classifier_only":
        # -------------------------------------------------------------
        # Train only the new classifier for NUM_EPOCHS, backbone frozen
        # -------------------------------------------------------------
        print("Training only the classifier for all epochs...")
        for param in model.ast_model.parameters():
            param.requires_grad = False

        # Ensure classifier is trainable
        for param in model.new_classifier.parameters():
            param.requires_grad = True
        
        optimizer = optim.Adam(model.new_classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        best_val_acc = 0.0
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']*100:.2f}%")
            
            scheduler.step(val_metrics['loss'])
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(model.state_dict(), SAVE_MODEL_PATH)
                print(f"Best classifier-only model saved with val accuracy: {best_val_acc*100:.2f}%")
    
    else:
        # -------------------------------------------------------------
        # "two_step" strategy:
        #   Stage 1 -> freeze backbone & train classifier (5 epochs)
        #   Stage 2 -> unfreeze backbone & fine-tune entire model
        # -------------------------------------------------------------
        
        # Stage 1: Train only the new classifier
        print("\nStage 1: Training only the classifier...")
        for param in model.ast_model.parameters():
            param.requires_grad = False
        
        for param in model.new_classifier.parameters():
            param.requires_grad = True
        
        optimizer = optim.Adam(model.new_classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        best_val_acc = 0.0
        for epoch in range(5):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/5 - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']*100:.2f}%")
            
            scheduler.step(val_metrics['loss'])
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(model.state_dict(), "ast_finetuned_stage1.pth")
                print(f"Stage 1 model saved with val accuracy: {best_val_acc*100:.2f}%")
        
        # Stage 2: Fine-tune the entire model
        print("\nStage 2: Fine-tuning the entire model...")
        for param in model.ast_model.parameters():
            param.requires_grad = True
        
        # Lower LR for backbone, normal LR for new classifier
        optimizer = optim.Adam([
            {'params': model.ast_model.parameters(), 'lr': LEARNING_RATE/10},
            {'params': model.new_classifier.parameters(), 'lr': LEARNING_RATE}
        ], weight_decay=WEIGHT_DECAY)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        best_val_acc = 0.0
        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']*100:.2f}%, "
                  f"Val F1: {val_metrics['f1']*100:.2f}%")
            
            scheduler.step(val_metrics['loss'])
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(model.state_dict(), SAVE_MODEL_PATH)
                print(f"Best model saved with val accuracy: {best_val_acc*100:.2f}%")
    
    # --------------------------------------------
    # Evaluate best model on the test set
    # --------------------------------------------
    print("\nEvaluating on the test set...")
    model.load_state_dict(torch.load(SAVE_MODEL_PATH))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_metrics = evaluate(model, test_loader, criterion, device)
    print("\nTest Results:")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print(f"Test Precision: {test_metrics['precision']*100:.2f}%")
    print(f"Test Recall: {test_metrics['recall']*100:.2f}%")
    print(f"Test F1 Score: {test_metrics['f1']*100:.2f}%")

if __name__ == "__main__":
    main()

