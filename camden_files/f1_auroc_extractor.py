import os
import pandas as pd
results_dir_path = "results"



f1 = []
auroc = []

sig_dict = {"EOG": [0, 1],
            "EEG": [2, 3],
            "RESP": [5, 6],
            "SPO2": [9],
            "CO2": [10],
            "ECG": [11, 12],
            "DEMO": [13],
            }

eog=[]
eeg=[]
resp=[]
sp2=[]
c2=[]
ecg=[]
demo=[]

for filename in os.listdir(results_dir_path):
    filepath = os.path.join(results_dir_path, filename)
    if os.path.isfile(filepath):
        with open(filepath, 'r') as file:
            for i, line in enumerate(file):
                if i == 13:
                    f1.append(float(line[4:8]))
                if i == 14:
                    auroc.append(float(line[7:12]))
                if i == 1:    
                    eog.append(1) if "EOG" in filename else eog.append(0)
                    eeg.append(1) if "EEG" in filename else eeg.append(0)
                    resp.append(1) if "RESP" in filename else resp.append(0)
                    sp2.append(1) if "SPO2" in filename else sp2.append(0)
                    ecg.append(1) if "ECG" in filename else ecg.append(0)
                    demo.append(1) if "DEMO" in filename else demo.append(0)
                    c2.append(1) if "CO2" in filename else c2.append(0)

df = pd.DataFrame({"EOG":eog,
                   "EEG":eeg,
                   "RESP":resp,
                   "SPO2":sp2,
                   "ECG":ecg,
                   "DEMO":demo,
                   "CO2":c2,
                   "F1":f1,
                   "ROC AUC":auroc
                   })

df.replace(0," ", inplace=True)
df.replace(1, "X", inplace=True)

df.to_csv(results_dir_path + "\\extracted.csv", index=False)