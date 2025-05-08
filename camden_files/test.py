import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import os
from metrics import Result
from models.instance_norm import InstanceNormalization
from models.transformer import PatchEncoder, Patches
# from data.noise_util import add_noise_to_data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

THRESHOLD = 1
FOLD = 5

def test(config, fold=None):
    x = []
    y_apnea = []
    y_hypopnea = []

    # Load data
    for file_name in sorted(os.listdir(config["data_path"])):
        if file_name.endswith('.npz'):
            file_path = os.path.join(config["data_path"], file_name)
            data = np.load(file_path, allow_pickle=True)
            if 'data' in data and 'labels_apnea' in data and 'labels_hypopnea' in data:
                x.append(data['data'])
                y_apnea.append(data['labels_apnea'])
                y_hypopnea.append(data['labels_hypopnea'])
            else:
                print(f"Skipping file {file_name} because it does not contain expected arrays.")

    if len(x) == 0:
        raise ValueError("No valid data files were found.")

    # Proper element-wise label combination
    y = [a + h for a, h in zip(y_apnea, y_hypopnea)]
    x = np.array(x, dtype=object)
    y = np.array(y, dtype=object)

    # Preprocessing
    for i in range(FOLD):
        x[i], y[i] = shuffle(x[i], y[i])
        x[i] = np.nan_to_num(x[i], nan=-1)
        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
        x[i] = x[i][:, :, config["channels"]]  # Channel selection

    result = Result()
    folds = range(FOLD) if fold is None else [fold]

    for fold in folds:
        x_test = x[fold]
        y_test = y[fold]
        

        model_path = config["model_path"] + str(fold) + ".keras"
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects={"InstanceNormalization": InstanceNormalization, "Patches": Patches, "PatchEncoder": PatchEncoder})
        predict = model.predict(x_test)
        y_score = predict
        y_predict = np.where(predict > 0.5, 1, 0)

        
        result.add(y_test, y_predict, y_score)

    result.print()
    os.makedirs("./results", exist_ok=True)
    result.save("./results/" + config["model_name"] + ".txt", config)

    del x_test, y_test, model, predict, y_score, y_predict


def test_age_seperated(config):
    x = []
    y_apnea = []
    y_hypopnea = []

    for i in range(10):
        data = np.load(config["data_path"] + str(i) + ".npz", allow_pickle=True)
        x.append(data['data'])
        y_apnea.append(data['labels_apnea'])
        y_hypopnea.append(data['labels_hypopnea'])

    y = [a + h for a, h in zip(y_apnea, y_hypopnea)]

    for i in range(10):
        x[i], y[i] = shuffle(x[i], y[i])
        x[i] = np.nan_to_num(x[i], nan=-1)
        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
        x[i] = x[i][:, :, config["channels"]]

    result = Result()

    for fold in range(10):
        x_test = x[fold]
        y_test = y[fold]

        model = tf.keras.models.load_model(config["model_path"] + "0.keras", compile=False)

        predict = model.predict(x_test)
        y_score = predict
        y_predict = np.where(predict > 0.5, 1, 0)

        result.add(y_test, y_predict, y_score)

    result.print()
    os.makedirs("./results", exist_ok=True)
    result.save("./results/" + config["model_name"] + ".txt", config)

    del x_test, y_test, model, predict, y_score, y_predict
