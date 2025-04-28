import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from metrics import Result
# from data.noise_util import add_noise_to_data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


THRESHOLD = 1
FOLD = 5


def test(config, fold=None):
    x = []
    y_apnea = []
    y_hypopnea = []

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

    x = np.array(x, dtype=object)
    y_apnea = np.array(y_apnea, dtype=object)
    y_hypopnea = np.array(y_hypopnea, dtype=object)
    y = y_apnea + y_hypopnea
    ############################################################################
    x, y_apnea, y_hypopnea = data['data'], data['y_apnea'], data['y_hypopnea']
    y = y_apnea + y_hypopnea
    for i in range(FOLD):
        x[i], y[i] = shuffle(x[i], y[i])
        x[i] = np.nan_to_num(x[i], nan=-1)
        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
        x[i] = x[i][:, :, config["channels"]]
    ############################################################################
    result = Result()
    folds = range(FOLD) if fold is None else [fold]
    for fold in folds:
        x_test = x[fold]
        # if config.get("test_noise_snr"):
        #     x_test = add_noise_to_data(x_test, config["test_noise_snr"])

        y_test = y[fold]  # For MultiClass keras.utils.to_categorical(y[fold], num_classes=2)

        model = tf.keras.models.load_model(config["model_path"] + str(fold), compile=False)

        predict = model.predict(x_test)
        y_score = predict
        y_predict = np.where(predict > 0.5, 1, 0)# For MultiClass np.argmax(y_score, axis=-1)

        result.add(y_test, y_predict, y_score)

    result.print()
    result.save("./results/" + config["model_name"] + ".txt", config)

    del data, x_test, y_test, model, predict, y_score, y_predict






def test_age_seperated(config):
    x = []
    y_apnea = []
    y_hypopnea = []
    for i in range(10):
        data = np.load(config["data_path"] + str(i) + ".npz", allow_pickle=True)
        x.append(data['data'])
        y_apnea.append(data['labels_apnea'])
        y_hypopnea.append(data['labels_hypopnea'])
    ############################################################################
    y = np.array(y_apnea) + np.array(y_hypopnea)
    for i in range(10):
        x[i], y[i] = shuffle(x[i], y[i])
        x[i] = np.nan_to_num(x[i], nan=-1)
        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
        x[i] = x[i][:, :, config["channels"]]
    ############################################################################
    result = Result()

    for fold in range(10):
        x_test = x[fold]
        # if config.get("test_noise_snr"):
        #     x_test = add_noise_to_data(x_test, config["test_noise_snr"])

        y_test = y[fold]  # For MultiClass keras.utils.to_categorical(y[fold], num_classes=2)

        model = tf.keras.models.load_model(config["model_path"] + str(0), compile=False)

        predict = model.predict(x_test)
        y_score = predict
        y_predict = np.where(predict > 0.5, 1, 0)# For MultiClass np.argmax(y_score, axis=-1)

        result.add(y_test, y_predict, y_score)

    result.print()
    result.save("./results/" + config["model_name"] + ".txt", config)

    del data, x_test, y_test, model, predict, y_score, y_predict