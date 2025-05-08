def load_data(path):
    root_dir = os.path.expanduser(path)
    file_list = os.listdir(root_dir)
    length = len(file_list)

    # Fold the data based on number of respiratory events
    study_event_counts = [i for i in range(0, length)]
    folds = []
    for i in range(5):
        folds.append(study_event_counts[i::5])

    x = []
    y_apnea = []
    y_hypopnea = []
    counter = 0
    for idx, fold in enumerate(folds):
        first = True
        for patient in fold:
            rri_succ_counter = 0
            rri_fail_counter = 0
            counter += 1
            print(counter)

            study_data = np.load(PATH + "\\" + file_list[patient - 1])

            signals = study_data['data']
            labels_apnea = study_data['labels_apnea']
            labels_hypopnea = study_data['labels_hypopnea']

            y_c = labels_apnea + labels_hypopnea
            neg_samples = np.where(y_c == 0)[0]
            pos_samples = list(np.where(y_c > 0)[0])
            ratio = len(pos_samples) / len(neg_samples)
            neg_survived = []
            for s in range(len(neg_samples)):
                if random.random() < ratio:
                    neg_survived.append(neg_samples[s])
            samples = neg_survived + pos_samples
            signals = signals[samples, :, :]
            labels_apnea = labels_apnea[samples]
            labels_hypopnea = labels_hypopnea[samples]

            data = np.zeros((signals.shape[0], EPOCH_LENGTH * FREQ, s_count + 2))
            for i in range(signals.shape[0]):  # for each epoch
                data[i, :, -1], data[i, :, -2], status = extract_rri(signals[i, ECG_SIG, :], FREQ,
                                                                     float(EPOCH_LENGTH))

                if status:
                    rri_succ_counter += 1
                else:
                    rri_fail_counter += 1

                for j in range(s_count):  # for each signal
                    data[i, :, j] = signals[i, SIGS[j], :]

            # Initialize aggregated variables
            if first:
                aggregated_data = data
                aggregated_label_apnea = labels_apnea
                aggregated_label_hypopnea = labels_hypopnea
                first = False
            else:
                aggregated_data = np.concatenate((aggregated_data, data), axis=0)
                aggregated_label_apnea = np.concatenate((aggregated_label_apnea, labels_apnea), axis=0)
                aggregated_label_hypopnea = np.concatenate((aggregated_label_hypopnea, labels_hypopnea), axis=0)

            print(rri_succ_counter, rri_fail_counter)

        # Append the aggregated data for the fold
        x.append(aggregated_data)
        y_apnea.append(aggregated_label_apnea)
        y_hypopnea.append(aggregated_label_hypopnea)

    return x, y_apnea, y_hypopnea
