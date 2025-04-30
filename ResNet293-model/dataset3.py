from scipy.signal import fftconvolve
from python_speech_features import sigproc
from torch.utils.data import Dataset
import scipy.io.wavfile as sciwav
import torch
import numpy as np
import pandas as pd
import torchaudio
import os

class WavDataset(Dataset):
    def __init__(self, csv_path, norm_type='std', max_duration=20.0, sample_rate=16000):
        """
        Args:
            csv_path: Path to the CSV manifest file with columns [file_path, speaker_id].
            norm_type: Normalization type ('std' or 'max').
            max_duration: Maximum duration in seconds for audio files.
            sample_rate: The sample rate to enforce for audio.
        """
        # Read CSV with header=0 to skip header if present
        self.df = pd.read_csv(csv_path, header=0)
        
        # Ensure column names match regardless of CSV header
        if "file_path" not in self.df.columns or "speaker_id" not in self.df.columns:
            self.df.columns = ["file_path", "speaker_id"]
            
        # Convert relative paths to absolute paths
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        self.df["file_path"] = self.df["file_path"].apply(
            lambda x: os.path.join(csv_dir, x) if not os.path.isabs(x) else x
        )
        
        self.norm_type = norm_type
        self.max_duration = max_duration
        self.sample_rate = sample_rate

        # Create speaker-to-id mapping
        unique_speakers = sorted(self.df["speaker_id"].unique())
        self.speaker2id = {spk: idx for idx, spk in enumerate(unique_speakers)}
        self.num_speakers = len(unique_speakers)
        
        # Verify files exist
        missing_files = self.df[~self.df["file_path"].apply(os.path.exists)]["file_path"].tolist()
        if missing_files:
            raise FileNotFoundError(f"Files not found: {missing_files[:5]}{'...' if len(missing_files) > 5 else ''}")

    def __len__(self):
        return len(self.df)

    def _load_data(self, file_path):
        """Load audio data from a given file path."""
        try:
            signal, sample_rate = torchaudio.load(file_path)

            # Resample if needed
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                signal = resampler(signal)

            # Apply duration limit (truncate to max_duration seconds)
            max_samples = int(self.max_duration * self.sample_rate)
            if signal.size(1) > max_samples:
                signal = signal[:, :max_samples]

            return signal.squeeze(0).numpy()  # Convert to 1D array if stereo
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {str(e)}")

    def _norm_speech(self, signal):
        """Normalize the speech signal."""
        if np.std(signal) == 0:
            return signal
        if self.norm_type == 'std':
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        elif self.norm_type == 'max':
            signal = signal / (np.abs(signal).max() + 1e-8)
        return signal

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        speaker_id = self.speaker2id[row["speaker_id"]]

        # Load and preprocess audio
        signal = self._load_data(file_path)
        signal = self._norm_speech(signal)
        signal = sigproc.preemphasis(signal, 0.97)
        signal = torch.tensor(signal, dtype=torch.float32)

        return signal, speaker_id

