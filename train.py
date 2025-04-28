import keras
import keras.metrics
import numpy as np
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.losses import BinaryCrossentropy
from sklearn.utils import shuffle
import os

from models.models import get_model

THRESHOLD = 1
FOLD = 5


def lr_schedule(epoch, lr):
    if epoch > 50 and (epoch - 1) % 5 == 0:
        lr *= 0.5
    return lr


def train(config, fold=None):
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
    ########################################################################################
    for i in range(FOLD):
        x[i], y[i] = shuffle(x[i], y[i])
        x[i] = np.nan_to_num(x[i], nan=-1)
        if config["regression"]:
            y[i] = np.sqrt(y[i])
            y[i][y[i] != 0] += 2
        else:
            y[i] = np.where(y[i] >= THRESHOLD, 1, 0)

        x[i] = x[i][:, :, config["channels"]]  # CHANNEL SELECTION

    ########################################################################################
    folds = range(FOLD) if fold is None else [fold]
    for fold in folds:
        first = True
        for i in range(5):
            if i != fold:
                if first:
                    x_train = x[i]
                    y_train = y[i]
                    first = False
                else:
                    x_train = np.concatenate((x_train, x[i]))
                    y_train = np.concatenate((y_train, y[i]))

        model = get_model(config)
        if config["regression"]:
            model.compile(optimizer="adam", loss=BinaryCrossentropy())
            early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        else:
            model.compile(optimizer="adam", loss=BinaryCrossentropy(),
                          metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
            early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        model.fit(x=x_train, y=y_train, batch_size=512, epochs=config["epochs"], validation_split=0.1,
                  callbacks=[early_stopper, lr_scheduler])
        ################################################################################################################
        model.save(config["model_path"] + str(fold))
        keras.backend.clear_session()


def train_age_seperated(config):
    data = np.load(config["data_path"], allow_pickle=True)
    x, y_apnea, y_hypopnea = data['data'], data['labels_apnea'], data['labels_hypopnea']
    y = y_apnea + y_hypopnea
    ########################################################################################
    for i in range(10):
        x[i], y[i] = shuffle(x[i], y[i])
        x[i] = np.nan_to_num(x[i], nan=-1)
        if config["regression"]:
            y[i] = np.sqrt(y[i])
            y[i][y[i] != 0] += 2
        else:
            y[i] = np.where(y[i] >= THRESHOLD, 1, 0)

        x[i] = x[i][:, :, config["channels"]]  # CHANNEL SELECTION

    ########################################################################################
    first = True
    for i in range(10):
        if first:
            x_train = x[i]
            y_train = y[i]
            first = False
        else:
            x_train = np.concatenate((x_train, x[i]))
            y_train = np.concatenate((y_train, y[i]))

    model = get_model(config)
    if config["regression"]:
        model.compile(optimizer="adam", loss=BinaryCrossentropy())
        early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    else:
        model.compile(optimizer="adam", loss=BinaryCrossentropy(),
                      metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
        early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    model.fit(x=x_train, y=y_train, batch_size=512, epochs=config["epochs"], validation_split=0.1,
              callbacks=[early_stopper, lr_scheduler])
    ################################################################################################################
    model.save(config["model_path"] + str(0))
    keras.backend.clear_session()
