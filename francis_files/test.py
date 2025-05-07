import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import os
from metrics import Result
from data.noise_util import add_noise_to_data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


THRESHOLD = 1
FOLD = 5


def test(config, fold=None):
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
                print(f"Loaded test data shape: {fold_data.shape}")
                
                # Extract the channels we need before transposing
                selected_channels = []
                for channel_idx in config["channels"]:
                    if channel_idx < fold_data.shape[1]:  # Make sure channel exists
                        selected_channels.append(fold_data[:, channel_idx, :])
                
                # Stack the selected channels along a new axis
                fold_data = np.stack(selected_channels, axis=1)
                print(f"Selected test channels shape: {fold_data.shape}")
                
                # Transpose the dimensions from (epochs, channels, samples) to (epochs, samples, channels)
                fold_data = np.transpose(fold_data, (0, 2, 1))
                print(f"Transposed test data shape: {fold_data.shape}")
                
                # Check if we need to downsample CHAT data (from 3840 to 1920 samples)
                if fold_data.shape[1] == 3840 and 'chat' in config["data_path"].lower():
                    print(f"Downsampling CHAT data from 3840 to 1920 samples per epoch")
                    # Downsample by taking every other sample
                    fold_data = fold_data[:, ::2, :]
                    print(f"New test shape after downsampling: {fold_data.shape}")
                
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
    
    # Process data for testing
    for i in range(len(x)):
        x[i], y[i] = shuffle(x[i], y[i])
        x[i] = np.nan_to_num(x[i], nan=-1)
        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
    
    ############################################################################
    # Test on the specified folds
    result = Result()
    folds = range(len(x)) if fold is None else [fold]
    
    for fold in folds:
        if fold >= len(x):
            print(f"Skipping fold {fold} because it's out of range (we only have {len(x)} folds)")
            continue
            
        x_test = x[fold]
        if config.get("test_noise_snr"):
            x_test = add_noise_to_data(x_test, config["test_noise_snr"])

        y_test = y[fold]

        model = tf.keras.models.load_model(config["model_path"] + str(fold), compile=False)

        predict = model.predict(x_test)
        y_score = predict
        y_predict = np.where(predict > 0.5, 1, 0)

        result.add(y_test, y_predict, y_score)

    result.print()
    result.save("./results/" + config["model_name"] + ".txt", config)

    del x_test, y_test, model, predict, y_score, y_predict


def test_age_seperated(config):
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
                    print(f"New test shape after downsampling: {fold_data.shape}")
                
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
    
    # Process data for testing
    for i in range(len(x)):
        x[i], y[i] = shuffle(x[i], y[i])
        x[i] = np.nan_to_num(x[i], nan=-1)
        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
    
    ############################################################################
    # Test all folds with the model from fold 0
    result = Result()

    for fold in range(len(x)):
        x_test = x[fold]
        if config.get("test_noise_snr"):
            x_test = add_noise_to_data(x_test, config["test_noise_snr"])

        y_test = y[fold]

        model = tf.keras.models.load_model(config["model_path"] + str(0), compile=False)

        predict = model.predict(x_test)
        y_score = predict
        y_predict = np.where(predict > 0.5, 1, 0)

        result.add(y_test, y_predict, y_score)

    result.print()
    result.save("./results/" + config["model_name"] + ".txt", config)

    del x_test, y_test, model, predict, y_score, y_predict