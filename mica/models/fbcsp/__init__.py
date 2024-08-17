import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from ...utils.torch import dataset_to_numpy
from .userDefFunc import f_logVar, my_function


def make_log_var(band: Tuple[int, int], band_order: int, labels, n_channels, sampling_rate: int, trials):
    n_samples_for_feat = 500
    class_0_labels = np.where(labels == 0)[0]
    n_class_0_labels = len(class_0_labels)
    class_1_labels = np.where(labels == 1)[0]
    n_class_1_labels = len(class_1_labels)
    n_trials = len(trials)

    sr = np.zeros([1, 1], dtype=int)
    sr[0, 0] = sampling_rate
    band = ((band / sr) * 2)[0]
    b, a = signal.butter(band_order, band, "bandpass", analog=False)
    raw_eeg_data = np.empty((n_trials, n_channels, n_samples_for_feat))

    # filter signals
    for i, sig in enumerate(trials):
        temp = signal.lfilter(b, a, sig, 1)
        raw_eeg_data[i] = temp[:, 250:]

    raw_eeg_data_class_0 = raw_eeg_data[class_0_labels, :, :]  # beta band data from class 1 trials
    raw_eeg_data_class_1 = raw_eeg_data[class_1_labels, :, :]  # beta band data from class 2 trials

    # combine trials in to one
    raw_eeg_data_class_0 = np.swapaxes(raw_eeg_data_class_0, 0, 1)
    raw_eeg_data_class_1 = np.swapaxes(raw_eeg_data_class_1, 0, 1)
    csp_raw_eeg_data_class_0 = np.reshape(
        raw_eeg_data_class_0, (n_channels, int(n_class_0_labels * n_samples_for_feat))
    )
    csp_raw_eeg_data_class_1 = np.reshape(
        raw_eeg_data_class_1, (n_channels, int(n_class_1_labels * n_samples_for_feat))
    )

    w_csp = my_function(csp_raw_eeg_data_class_0, csp_raw_eeg_data_class_1)

    log_var = np.empty((n_trials, n_channels))

    for i, sig in enumerate(raw_eeg_data):  # USE FILTERED
        log_var[i] = f_logVar(
            np.matmul(w_csp, sig)
        )  # calculating the Z matrix for Beta band -> calculating the logvariance for Beta

    return log_var, w_csp, band, b, a


def make_feature_set(trials, feats):
    n_trials = len(trials)
    feat = np.empty((n_trials, len(feats) * 2))

    for i, log_vars in enumerate(zip(*[item[0] for item in feats])):
        feat[i] = [element for sublist in [[item[0], item[-1]] for item in log_vars] for element in sublist]

    return feat


def train_model(
    bands: List[Tuple[int, int]],
    sampling_rate: int,
    test_sets,
    train_set,
    plot_feature_space: bool = True,
    report_file_base: Path = None,
    subject_no: int = None,
):
    """
    Train CSP model using the trial data

    parameters:
        trials: 3d np array containing the trial EEG data with dimensions (x,y,z) where x=trials number,
        y=channel number, z=sample number
        outputfilname: path and name of the file where the trained model will be pickled and saved
    """
    cv_report_file = ho_report_file = None

    if report_file_base and subject_no:
        report_file_base.mkdir(parents=True, exist_ok=True)

        cv_report_file = report_file_base / f"report_cv_P{subject_no:03d}.json"
        ho_report_file = report_file_base / f"report_ho_P{subject_no:03d}.json"

    train_trials, train_labels = dataset_to_numpy(train_set)
    n_channels = train_trials.shape[1]  # getting the number of channels

    train_feats = []

    for band in bands:
        train_feats.append(make_log_var(band, 2, train_labels, n_channels, sampling_rate, train_trials))

    X_train = make_feature_set(train_trials, train_feats)
    y_train = train_labels
    print("X_train.shape", X_train.shape)
    print("y_train.shape", y_train.shape)

    # Fit model
    clf = svm.SVC(kernel="linear", C=1)
    model = clf.fit(X_train, y_train)

    # Eval (CV)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    cv_mean_acc = np.mean(scores)
    if cv_report_file:
        cv_report_file.write_text(
            json.dumps(
                [
                    {
                        "acc_avg": cv_mean_acc,
                        "acc_std": np.std(scores),
                    }
                ],
                indent=4,
            )
        )
    print("cv acc:", cv_mean_acc)

    # Eval (All)
    y_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_preds)
    print("train acc:", train_acc)

    # Eval (Test)
    test_reports = [
        {
            "eval_report": {
                "acc": train_acc,
            },
            "window": "TRAIN_SET_EVAL",
        },
    ]
    for test_set in test_sets:
        test_trials, test_labels = dataset_to_numpy(test_set["dataset"])
        test_feats = []

        for band in bands:
            test_feats.append(make_log_var(band, 2, test_labels, n_channels, sampling_rate, test_trials))

        X_test = make_feature_set(test_trials, test_feats)
        y_test = test_labels
        y_preds = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_preds)
        test_reports.append(
            {
                "eval_report": {
                    "acc": test_acc,
                    "y_pred": y_preds.tolist(),
                    "y_true": y_test.tolist(),
                },
                "window": test_set["window"],
            }
        )
        print(f"{test_set['window']} test acc:", test_acc)

    if ho_report_file:
        ho_report_file.write_text(json.dumps(test_reports, indent=4))

    # plot feature space
    if plot_feature_space:
        correct_class1 = []
        correct_class1_lbl = []
        wrong_class1 = []
        wrong_class1_lbl = []
        correct_class2 = []
        correct_class2_lbl = []
        wrong_clas2 = []
        wrong_class2_lbl = []
        for trx, pred, lbl in zip(X_test, y_preds, y_test):
            if pred == lbl:  # correct prediction
                if lbl == 1:
                    correct_class1.append(trx)
                    correct_class1_lbl.append(lbl)
                else:
                    correct_class2.append(trx)
                    correct_class2_lbl.append(lbl)
            else:  # wrong prediction
                if lbl == 1:
                    wrong_class1.append(trx)
                    wrong_class1_lbl.append(lbl)
                else:
                    wrong_clas2.append(trx)
                    wrong_class2_lbl.append(lbl)

        correct_class1 = np.array(correct_class1)
        correct_class2 = np.array(correct_class2)
        wrong_class1 = np.array(wrong_class1)
        wrong_clas2 = np.array(wrong_clas2)

        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.set_title("Alpha")
        grphs_a = [0] * 4

        if correct_class1.any():
            grphs_a[0] = ax1.scatter(correct_class1[:, 0], correct_class1[:, 1], c="b", marker="o")
            grphs_a[0].set_label("cl1 correct")
        if wrong_class1.any():
            grphs_a[1] = ax1.scatter(wrong_class1[:, 0], wrong_class1[:, 1], c="b", marker="X")
            grphs_a[1].set_label("cl1 wrong")
        if correct_class2.any():
            grphs_a[2] = ax1.scatter(correct_class2[:, 0], correct_class2[:, 1], c="g", marker="o")
            grphs_a[2].set_label("cl2 correct")
        if wrong_clas2.any():
            grphs_a[3] = ax1.scatter(wrong_clas2[:, 0], wrong_clas2[:, 1], c="g", marker="X")
            grphs_a[3].set_label("cl2 wrong")
        ax1.legend()

        ax2.set_title("Beta")
        grphs_b = [0] * 4
        if correct_class1.any():
            grphs_b[0] = ax2.scatter(correct_class1[:, 2], correct_class1[:, 3], c="b", marker="o")
            grphs_b[0].set_label("cl1 correct")
        if wrong_class1.any():
            grphs_b[1] = ax2.scatter(wrong_class1[:, 2], wrong_class1[:, 3], c="b", marker="X")
            grphs_b[1].set_label("cl1 wrong")
        if correct_class2.any():
            grphs_b[2] = ax2.scatter(correct_class2[:, 2], correct_class2[:, 3], c="g", marker="o")
            grphs_b[2].set_label("cl2 correct")
        if wrong_clas2.any():
            grphs_b[3] = ax2.scatter(wrong_clas2[:, 2], wrong_clas2[:, 3], c="g", marker="X")
            grphs_b[3].set_label("cl2 wrong")
        ax2.legend()

        fig.text(0.5, 0.01, "meanCVacc: %0.2f  trainAcc: %0.2f" % (cv_mean_acc, train_acc), ha="center", size=20)
        plt.show()
