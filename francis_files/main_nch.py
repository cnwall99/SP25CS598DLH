import gc

from test import test
from train import train

# "EOG LOC-M2",  # 0
# "EOG ROC-M1",  # 1
# "EEG C3-M2",  # 2
# "EEG C4-M1",  # 3
# "ECG EKG2-EKG",  # 4
#
# "RESP PTAF",  # 5
# "RESP AIRFLOW",  # 6
# "RESP THORACIC",  # 7
# "RESP ABDOMINAL",  # 8
# "SPO2",  # 9
# "CAPNO",  # 10

######### ADDED IN THIS STEP #########
# RRI #11
# Ramp #12
# Demo #13


sig_dict = {"EOG": [0, 1],
            "EEG": [2, 3],
            "RESP": [5, 6],
            "SPO2": [9],
            "CO2": [10],
            "ECG": [11, 12],
            "DEMO": [13],
            }

channel_list = [

    ["ECG", "SPO2"],

]

for ch in channel_list:
    chs = []
    chstr = ""
    for name in ch:
        chstr += name
        chs = chs + sig_dict[name]
    config = {
        "data_path": "C:\\Users\\Francis\\Desktop\\Pediatric-Apnea-Detection-main\\data\\nch\\",
        "model_path": "C:\\Users\\Francis\\Desktop\\Pediatric-Apnea-Detection-main\\weights\\semscnn_ecgspo2\\f",
        "model_name": "sem-mscnn_" + chstr,
        "regression": False,

        # Add loss function parameters
        "loss_type": "bce",  # Options: "bce", "focal", "combined"
        "focal_gamma": 2.0,
        "focal_alpha": 0.25,
        "bce_weight": 0.5,
        "focal_weight": 0.5,

        "transformer_layers": 5,  # best 5
        "drop_out_rate": 0.25,  # best 0.25
        "num_patches": 30,  # best 30 TBD
        "transformer_units": 32,  # best 32
        "regularization_weight": 0.001,  # best 0.001
        "num_heads": 4,
        "epochs": 100,  # best 200
        "channels": chs,
    }
    train(config, 0)
    test(config, 0)
    gc.collect()
