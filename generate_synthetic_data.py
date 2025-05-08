
import numpy as np
import os
import random

# Paths
OUTPUT_CHAT_PATH = "Pediatric-Apnea-Detection-main\\data\\chat"
OUTPUT_NCH_PATH = "Pediatric-Apnea-Detection-main\\data\\nch"

# Constants
CHAT_FREQ = 128
NCH_FREQ = 64
EPOCH_LENGTH = 30  # seconds

# Number of samples to generate
NUM_EPOCHS_PER_FOLD = 1000  # Adjust as needed
NUM_FOLDS = 5

def generate_synthetic_signal(n_samples, freq_components=[1, 2, 5, 10], noise_level=0.2):
    """Generate a synthetic signal with multiple frequency components and noise"""
    t = np.linspace(0, EPOCH_LENGTH, n_samples)
    signal = np.zeros(n_samples)
    
    # Add several sine waves of different frequencies
    for freq in freq_components:
        amplitude = random.uniform(0.5, 2.0)
        phase = random.uniform(0, 2*np.pi)
        signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # Add random noise
    noise = np.random.normal(0, noise_level, n_samples)
    signal += noise
    
    # Normalize
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    
    return signal

def create_synthetic_chat_data():
    """Create synthetic CHAT dataset"""
    print("Generating synthetic CHAT data...")
    
    # Create output directory
    os.makedirs(OUTPUT_CHAT_PATH, exist_ok=True)
    
    n_samples = CHAT_FREQ * EPOCH_LENGTH  # 3840 samples per epoch
    n_channels = 17  # Full number of channels for CHAT dataset
    
    for fold in range(NUM_FOLDS):
        # Create a fold of data
        fold_data = np.zeros((NUM_EPOCHS_PER_FOLD, n_channels, n_samples))  # Note: channels before samples
        fold_apnea = np.zeros(NUM_EPOCHS_PER_FOLD)
        fold_hypopnea = np.zeros(NUM_EPOCHS_PER_FOLD)
        
        for i in range(NUM_EPOCHS_PER_FOLD):
            # Set some epochs to have apnea or hypopnea events
            if random.random() < 0.15:  # 30% apnea events
                fold_apnea[i] = random.randint(3, 15)  # Duration in seconds
            elif random.random() < 0.20:  # 30% hypopnea events
                fold_hypopnea[i] = random.randint(3, 15)  # Duration in seconds
            
            # Generate data for all channels
            for j in range(n_channels):
                # Different frequency components for different types of signals
                if j < 8:  # EEG/EOG channels
                    freqs = [1, 4, 8, 12, 20]
                elif j == 8:  # ECG channel
                    freqs = [1, 1.2, 1.5]
                elif j < 13:  # Respiratory channels
                    freqs = [0.2, 0.3, 0.5]
                elif j == 13:  # SAO2 (SpO2)
                    freqs = [0.1, 0.2]
                elif j == 14:  # CAP
                    freqs = [0.3, 0.5, 1]
                elif j == 15 or j == 16:  # RRI and Ramp (ECG derived)
                    freqs = [1, 1.2, 1.5]
                else:
                    freqs = [0.5, 1, 2]
                
                # Generate the signal
                fold_data[i, j, :] = generate_synthetic_signal(n_samples, freqs)
            
            # Add features for apnea/hypopnea events
            if fold_apnea[i] > 0 or fold_hypopnea[i] > 0:
                # Event duration in samples
                event_duration = int((fold_apnea[i] + fold_hypopnea[i]) * CHAT_FREQ)
                start_idx = random.randint(0, n_samples - event_duration - 1)
                
                # Modify respiratory signals
                for j in range(9, 13):  # Respiratory channels
                    if fold_apnea[i] > 0:  # Apnea - stronger signal reduction
                        fold_data[i, j, start_idx:start_idx+event_duration] *= 0.1
                    else:  # Hypopnea - moderate signal reduction
                        fold_data[i, j, start_idx:start_idx+event_duration] *= 0.5
                
                # Add SpO2 desaturation (with delay)
                desat_start = min(n_samples - event_duration - 1, 
                                 start_idx + int(CHAT_FREQ * 5))
                
                # Create desaturation pattern
                desaturation = np.linspace(0, -1, event_duration // 2)
                recovery = np.linspace(-1, 0, event_duration - event_duration // 2)
                desat_pattern = np.concatenate([desaturation, recovery])
                
                # Apply pattern to SpO2 signal
                fold_data[i, 13, desat_start:desat_start+event_duration] += desat_pattern
        
        # Save individual fold files with the expected array names
        np.savez_compressed(
            os.path.join(OUTPUT_CHAT_PATH, f"chat_fold_{fold}.npz"),
            data=fold_data,
            labels_apnea=fold_apnea,
            labels_hypopnea=fold_hypopnea
        )
        print(f"Saved CHAT fold {fold} with shape {fold_data.shape}")
    
    print(f"Generated {NUM_FOLDS} CHAT data files")

def create_synthetic_nch_data():
    """Create synthetic NCH dataset"""
    print("Generating synthetic NCH data...")
    
    # Create output directory
    os.makedirs(OUTPUT_NCH_PATH, exist_ok=True)
    
    n_samples = NCH_FREQ * EPOCH_LENGTH  # 1920 samples per epoch
    n_channels = 14  # Full number of channels for NCH dataset
    
    for fold in range(NUM_FOLDS):
        # Create a fold of data
        fold_data = np.zeros((NUM_EPOCHS_PER_FOLD, n_channels, n_samples))  # Note: channels before samples
        fold_apnea = np.zeros(NUM_EPOCHS_PER_FOLD)
        fold_hypopnea = np.zeros(NUM_EPOCHS_PER_FOLD)
        
        for i in range(NUM_EPOCHS_PER_FOLD):
            # Set some epochs to have apnea or hypopnea events
            if random.random() < 0.15:  # 30% apnea events
                fold_apnea[i] = random.randint(3, 15)  # Duration in seconds
            elif random.random() < 0.20:  # 30% hypopnea events
                fold_hypopnea[i] = random.randint(3, 15)  # Duration in seconds
            
            # Generate data for all channels
            for j in range(n_channels):
                # Different frequency components for different types of signals
                if j < 4:  # EEG/EOG channels
                    freqs = [1, 4, 8, 12, 20]
                elif j == 4:  # ECG channel
                    freqs = [1, 1.2, 1.5]
                elif j < 9:  # Respiratory channels
                    freqs = [0.2, 0.3, 0.5]
                elif j == 9:  # SpO2
                    freqs = [0.1, 0.2]
                elif j == 10:  # CAPNO
                    freqs = [0.3, 0.5, 1]
                elif j == 11 or j == 12:  # RRI and Ramp (ECG derived)
                    freqs = [1, 1.2, 1.5]
                else:  # Demographic info
                    # For demographic channel, use a constant value
                    fold_data[i, j, :] = np.ones(n_samples) * random.uniform(-1, 1)
                    continue
                
                # Generate the signal
                fold_data[i, j, :] = generate_synthetic_signal(n_samples, freqs)
            
            # Add features for apnea/hypopnea events
            if fold_apnea[i] > 0 or fold_hypopnea[i] > 0:
                # Event duration in samples
                event_duration = int((fold_apnea[i] + fold_hypopnea[i]) * NCH_FREQ)
                start_idx = random.randint(0, n_samples - event_duration - 1)
                
                # Modify respiratory signals
                for j in range(5, 9):  # Respiratory channels
                    if fold_apnea[i] > 0:  # Apnea - stronger signal reduction
                        fold_data[i, j, start_idx:start_idx+event_duration] *= 0.1
                    else:  # Hypopnea - moderate signal reduction
                        fold_data[i, j, start_idx:start_idx+event_duration] *= 0.5
                
                # Add SpO2 desaturation (with delay)
                desat_start = min(n_samples - event_duration - 1, 
                                 start_idx + int(NCH_FREQ * 5))
                
                # Create desaturation pattern
                desaturation = np.linspace(0, -1, event_duration // 2)
                recovery = np.linspace(-1, 0, event_duration - event_duration // 2)
                desat_pattern = np.concatenate([desaturation, recovery])
                
                # Apply pattern to SpO2 signal
                fold_data[i, 9, desat_start:desat_start+event_duration] += desat_pattern
        
        # Save individual fold files with the expected array names
        np.savez_compressed(
            os.path.join(OUTPUT_NCH_PATH, f"nch_fold_{fold}.npz"),
            data=fold_data,
            labels_apnea=fold_apnea,
            labels_hypopnea=fold_hypopnea
        )
        print(f"Saved NCH fold {fold} with shape {fold_data.shape}")
    
    print(f"Generated {NUM_FOLDS} NCH data files")

def main():
    create_synthetic_chat_data()
    create_synthetic_nch_data()
    print("Data generation complete!")

if __name__ == "__main__":
    main()

