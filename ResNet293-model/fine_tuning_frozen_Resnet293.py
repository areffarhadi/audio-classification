import torch
import torch.nn as nn
import torchaudio
import logging
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from hyperpyyaml import load_hyperpyyaml
from dataset3 import WavDataset
from feat import logFbankCal
from optim import Eden
import os

# Collate function for data loader
def collate_fn(batch):
    signals = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    max_len = max(signal.size(0) for signal in signals)
    padded_signals = torch.zeros(len(signals), max_len)
    for i, signal in enumerate(signals):
        padded_signals[i, :signal.size(0)] = signal
    return padded_signals, torch.tensor(targets)

# Model definition
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

# Accuracy computation
def compute_accuracy(predictions, targets):
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total * 100

# Training function
from tqdm import tqdm  # Add this import at the top

# Training function
def train_model(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = load_hyperpyyaml(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    os.environ["WANDB_MODE"] = "disabled"
    
    base_model = cfg['model']
    state_dict = torch.load(cfg['ckpt_path'], map_location=device)
    base_model.load_state_dict(state_dict, strict=False)
    
    train_dataset = WavDataset(cfg['train_manifest'], norm_type=cfg['norm_type'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=cfg['num_workers'],
        collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        WavDataset(cfg['valid_manifest'], norm_type=cfg['norm_type']),
        batch_size=8,
        shuffle=False,
        num_workers=cfg['num_workers'],
        collate_fn=collate_fn
    )
    
    model = FineTunedSpeakerModel(
        base_model=base_model,
        num_speakers=train_dataset.num_speakers,
        embd_dim=256
    ).to(device)
    
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=cfg['base_lr'])
    scheduler = Eden(optimizer, cfg['lr_batches'], cfg['lr_epochs'], 
                    warmup_batches=cfg['warmup_batches'])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg['use_fp16'])
    
    best_val_loss = float('inf')
    
    for epoch in range(cfg['num_epochs']):
        model.train()
        epoch_correct = 0
        epoch_total = 0
        
        # Add tqdm progress bar for training
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch} [Training]", unit="batch")
        
        for i, (inputs, targets) in enumerate(train_progress):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            with autocast(enabled=cfg['use_fp16']):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step_batch(i)
            
            # Compute batch accuracy
            predictions = torch.argmax(outputs, dim=1)
            batch_correct = (predictions == targets).sum().item()
            batch_total = targets.size(0)
            batch_accuracy = batch_correct / batch_total * 100
            
            epoch_correct += batch_correct
            epoch_total += batch_total
            
            # Update progress bar
            train_progress.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Accuracy": f"{batch_accuracy:.2f}%"
            })

        epoch_accuracy = epoch_correct / epoch_total * 100
        logging.info(f"Epoch {epoch}: Training Accuracy: {epoch_accuracy:.2f}%")
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Add tqdm progress bar for validation
        val_progress = tqdm(valid_loader, desc=f"Epoch {epoch} [Validation]", unit="batch")
        
        with torch.no_grad():
            for inputs, targets in val_progress:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                predictions = torch.argmax(outputs, dim=1)
                val_correct += (predictions == targets).sum().item()
                val_total += targets.size(0)
                
                # Update validation progress bar
                val_progress.set_postfix({
                    "Loss": f"{val_loss / len(valid_loader):.4f}",
                    "Accuracy": f"{(val_correct / val_total) * 100:.2f}%"
                })
        
        val_loss /= len(valid_loader)
        val_accuracy = val_correct / val_total * 100
        logging.info(f"Epoch {epoch}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'speaker2id': train_dataset.speaker2id,
                'architecture': model
            }, f"{cfg['exp_dir']}/best_model4.pth")
            torch.save(model.state_dict(), f"{cfg['exp_dir']}/base_model4.pt")
            logging.info(f"Saved new best model with val_loss: {val_loss:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model('./conf/resnet293_4.yaml')

