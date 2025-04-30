import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader
from dataset3 import WavDataset
from tqdm import tqdm

# Include the model definition for proper loading
class FineTunedSpeakerModel(nn.Module):
    def __init__(self, base_model, num_speakers, embd_dim=256):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(embd_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_speakers)
        )
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        embeddings = self.extract_embeddings(x)
        return self.classifier(embeddings)
    
    def extract_embeddings(self, x):
        with torch.no_grad():
            return self.base_model(x)

def collate_fn(batch):
    signals = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    max_len = max(signal.size(0) for signal in signals)
    padded_signals = torch.zeros(len(signals), max_len)
    for i, signal in enumerate(signals):
        padded_signals[i, :signal.size(0)] = signal
    return padded_signals, torch.tensor(targets)

def test_model(model_path, test_manifest, batch_size=8, num_workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the saved model
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['architecture']
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test dataset and loader
    test_dataset = WavDataset(test_manifest, norm_type='mean-norm')
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    # Create progress bar
    test_progress = tqdm(test_loader, desc="Testing", unit="batch")
    
    with torch.no_grad():
        for inputs, targets in test_progress:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            
            # Update accuracy metrics
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            
            # Update progress bar with current accuracy
            current_accuracy = (correct / total) * 100
            test_progress.set_postfix({
                "Accuracy": f"{current_accuracy:.2f}%"
            })
    
    final_accuracy = (correct / total) * 100
    print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")
    return final_accuracy

if __name__ == "__main__":
    MODEL_PATH = "./voice_res293_ident/best_model4.pth"  # Update this path
    TEST_MANIFEST = "wav_manifest_test2.csv"  # Update this path
    
    accuracy = test_model(MODEL_PATH, TEST_MANIFEST)
