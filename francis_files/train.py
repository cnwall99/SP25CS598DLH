import os
import keras
import keras.metrics
import numpy as np
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.losses import BinaryCrossentropy
from losses import FocalLoss, CombinedBCEFocalLoss
from sklearn.utils import shuffle
import tensorflow as tf

from models.models import get_model

THRESHOLD = 1
FOLD = 5


def lr_schedule(epoch, lr):
    if epoch > 50 and (epoch - 1) % 5 == 0:
        lr *= 0.5
    return lr

def get_loss_function(config):
    """Get the appropriate loss function based on configuration."""
    loss_type = config.get("loss_type", "bce")
    
    if loss_type == "focal":
        return FocalLoss(
            gamma=config.get("focal_gamma", 2.0),
            alpha=config.get("focal_alpha", 0.25)
        )
    elif loss_type == "combined":
        return CombinedBCEFocalLoss(
            gamma=config.get("focal_gamma", 2.0),
            alpha=config.get("focal_alpha", 0.25),
            bce_weight=config.get("bce_weight", 0.5),
            focal_weight=config.get("focal_weight", 0.5)
        )
    else:  # Default to BCE
        return BinaryCrossentropy()

def train(config, fold=None):
    # Initialize lists to store data from multiple files
    x = []
    y_apnea = []
    y_hypopnea = []

    # Load data from all NPZ files in the directory
    for file_name in sorted(os.listdir(config["data_path"])):
        if file_name.endswith('.npz'):
            file_path = os.path.join(config["data_path"], file_name)
            data = np.load(file_path, allow_pickle=True)
            
            # Check if the file contains the expected arrays
            if 'data' in data and 'labels_apnea' in data and 'labels_hypopnea' in data:
                # Get the data and transpose the dimensions to what the model expects
                fold_data = data['data']
                print(f"Loaded data shape: {fold_data.shape}")
                
                # Extract the channels we need before transposing
                selected_channels = []
                for channel_idx in config["channels"]:
                    if channel_idx < fold_data.shape[1]:  # Make sure channel exists
                        selected_channels.append(fold_data[:, channel_idx, :])
                
                # Stack the selected channels along a new axis
                fold_data = np.stack(selected_channels, axis=1)
                print(f"Selected channels shape: {fold_data.shape}")
                
                # Transpose the dimensions from (epochs, channels, samples) to (epochs, samples, channels)
                fold_data = np.transpose(fold_data, (0, 2, 1))
                print(f"Transposed data shape: {fold_data.shape}")
                
                # Check if we need to downsample CHAT data (from 3840 to 1920 samples)
                if fold_data.shape[1] == 3840 and 'chat' in config["data_path"].lower():
                    print(f"Downsampling CHAT data from 3840 to 1920 samples per epoch")
                    # Downsample by taking every other sample
                    fold_data = fold_data[:, ::2, :]
                    print(f"New shape after downsampling: {fold_data.shape}")
                
                # Ensure the data is 32-bit float
                fold_data = fold_data.astype(np.float32)
                
                x.append(fold_data)
                y_apnea.append(data['labels_apnea'])
                y_hypopnea.append(data['labels_hypopnea'])
            else:
                print(f"Skipping file {file_name} because it does not contain expected arrays.")

    # Check if any valid data was loaded
    if len(x) == 0:
        raise ValueError("No valid data files were found.")
    
    # Initialize the y list by combining apnea and hypopnea labels
    y = []
    for i in range(len(y_apnea)):
        y.append(y_apnea[i] + y_hypopnea[i])
    
    ########################################################################################
    # Process each fold
    for i in range(len(x)):
        # Print shape for debugging
        print(f"Fold {i} shape before processing: {x[i].shape}")
        
        # Shuffle data and labels
        x[i], y[i] = shuffle(x[i], y[i])
        
        # Replace NaN values with -1
        x[i] = np.nan_to_num(x[i], nan=-1)
        
        # Process labels based on regression flag
        if config["regression"]:
            y[i] = np.sqrt(y[i])
            y[i][y[i] != 0] += 2
        else:
            y[i] = np.where(y[i] >= THRESHOLD, 1, 0)

        # Make sure labels are float32
        y[i] = y[i].astype(np.float32)
        
        print(f"Fold {i} final shape: {x[i].shape}")

    ########################################################################################
    # Determine which folds to train on
    folds = range(len(x)) if fold is None else [fold]
    
    for fold in folds:
        if fold >= len(x):
            print(f"Skipping fold {fold} because it's out of range (we only have {len(x)} folds)")
            continue
            
        # Use all folds except the current one for training
        first = True
        for i in range(len(x)):
            if i != fold:
                if first:
                    x_train = x[i]
                    y_train = y[i]
                    first = False
                else:
                    x_train = np.concatenate((x_train, x[i]))
                    y_train = np.concatenate((y_train, y[i]))

        # Debug: Check shapes and types
        print(f"x_train shape: {x_train.shape}, type: {x_train.dtype}")
        print(f"y_train shape: {y_train.shape}, type: {y_train.dtype}")
        
        # Ensure they're proper tensors
        x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
        
        # Create and compile the model
        model = get_model(config)
        loss_fn = get_loss_function(config)
        if config["regression"]:
            model.compile(optimizer="adam", loss=loss_fn)
            early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        else:
            model.compile(optimizer="adam", loss=loss_fn,
                          metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
            early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Set up learning rate scheduler
        lr_scheduler = LearningRateScheduler(lr_schedule)
        
        # Train the model using the tensors
        model.fit(x=x_train_tensor, y=y_train_tensor, batch_size=512, epochs=config["epochs"], 
                  validation_split=0.1, callbacks=[early_stopper, lr_scheduler])
        
        ################################################################################################################
        # Save the trained model
        model.save(config["model_path"] + str(fold))
        keras.backend.clear_session()


def train_age_seperated(config):
    # Initialize lists to store data from multiple files
    x = []
    y_apnea = []
    y_hypopnea = []

    # Load data from all NPZ files in the directory
    for file_name in sorted(os.listdir(config["data_path"])):
        if file_name.endswith('.npz'):
            file_path = os.path.join(config["data_path"], file_name)
            data = np.load(file_path, allow_pickle=True)
            
            # Check if the file contains the expected arrays
            if 'data' in data and 'labels_apnea' in data and 'labels_hypopnea' in data:
                # Get the data and select only the channels we need
                fold_data = data['data']
                
                # Extract the channels we need before transposing
                selected_channels = []
                for channel_idx in config["channels"]:
                    if channel_idx < fold_data.shape[1]:  # Make sure channel exists
                        selected_channels.append(fold_data[:, channel_idx, :])
                
                # Stack the selected channels along a new axis
                fold_data = np.stack(selected_channels, axis=1)
                
                # Transpose the dimensions from (epochs, channels, samples) to (epochs, samples, channels)
                fold_data = np.transpose(fold_data, (0, 2, 1))
                
                # Check if we need to downsample CHAT data (from 3840 to 1920 samples)
                if fold_data.shape[1] == 3840 and 'chat' in config["data_path"].lower():
                    print(f"Downsampling CHAT data from 3840 to 1920 samples per epoch")
                    # Downsample by taking every other sample
                    fold_data = fold_data[:, ::2, :]
                    print(f"New shape after downsampling: {fold_data.shape}")
                
                # Ensure the data is 32-bit float
                fold_data = fold_data.astype(np.float32)
                
                x.append(fold_data)
                y_apnea.append(data['labels_apnea'])
                y_hypopnea.append(data['labels_hypopnea'])
            else:
                print(f"Skipping file {file_name} because it does not contain expected arrays.")

    # Check if any valid data was loaded
    if len(x) == 0:
        raise ValueError("No valid data files were found.")
    
    # Initialize the y list by combining apnea and hypopnea labels
    y = []
    for i in range(len(y_apnea)):
        y.append(y_apnea[i] + y_hypopnea[i])
    
    ########################################################################################
    # Process each fold (up to 10)
    for i in range(min(10, len(x))):
        # Shuffle data and labels
        x[i], y[i] = shuffle(x[i], y[i])
        
        # Replace NaN values with -1
        x[i] = np.nan_to_num(x[i], nan=-1)
        
        # Process labels based on regression flag
        if config["regression"]:
            y[i] = np.sqrt(y[i])
            y[i][y[i] != 0] += 2
        else:
            y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
            
        # Make sure labels are float32
        y[i] = y[i].astype(np.float32)

    ########################################################################################
    # Combine all folds for training
    first = True
    for i in range(min(10, len(x))):
        if first:
            x_train = x[i]
            y_train = y[i]
            first = False
        else:
            x_train = np.concatenate((x_train, x[i]))
            y_train = np.concatenate((y_train, y[i]))

    # Convert to tensors
    x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
    
    # Create and compile the model
    model = get_model(config)
    loss_fn = get_loss_function(config)
    if config["regression"]:
        model.compile(optimizer="adam", loss=loss_fn)
        early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    else:
        model.compile(optimizer="adam", loss=loss_fn,
                      metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
        early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Set up learning rate scheduler
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    # Train the model
    model.fit(x=x_train_tensor, y=y_train_tensor, batch_size=512, epochs=config["epochs"], 
              validation_split=0.1, callbacks=[early_stopper, lr_scheduler])
    
    ################################################################################################################
    # Save the trained model
    model.save(config["model_path"] + str(0))
    keras.backend.clear_session()