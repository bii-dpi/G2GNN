import os
import pickle
import numpy as np
import pandas as pd
from progressbar import progressbar
from sklearn.metrics import (roc_auc_score,
                             precision_score,
                             recall_score,
                             average_precision_score,
                             precision_recall_curve)
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor as PPE


def get_data(direction,mode, mode_):
    data = pd.read_csv(f"../../get_data/NewData/results/csv/"
                       f"{direction}_testing_examples{mode}.csv")

    labels = data.Y
    if mode_ == "_ifp":
        features = data.drop(columns="Y")
    else:
        features = data.drop(columns="Y").iloc[:, :79]

    return np.array(features.to_numpy(), dtype=float), np.array(labels.to_numpy(), dtype=int)


def get_recall(precisions, recalls, prec_val):
    precisions = np.abs(np.array(precisions) - prec_val)

    return recalls[np.argmin(precisions)]


def get_ef(y, prec_val, num_pos):
    len_to_take = int(len(y) * prec_val)

    return (np.sum(y[:len_to_take]) / num_pos) * 100


def get_log_auc(predicted, y, num_pos):
    prec_vals = np.linspace(.0001, 1, 10000)
    recalls = []
    for prec_val in prec_vals:
        recalls.append(float(get_ef(y, prec_val, num_pos)))

    return np.trapz(y=recalls, x=np.log10(prec_vals) / 3, dx=1/30)


def get_performance(pair):
    direction, mode, mode_ = pair
    model = pd.read_pickle(f"../models/{direction}{mode}{mode_}.pkl")
    X, y = get_data(direction,mode, mode_)

    predictions = model.predict_proba(X)[:, 1]

    pdb_ids = \
        np.load(f"../../get_data/NewData/results/features/{direction}_testing_pdb_ids.npy",
                      allow_pickle=True)

    prediction_dict = defaultdict(list)
    y_dict = defaultdict(list)
    for i in range(len(pdb_ids)):
        prediction_dict[pdb_ids[i]].append(predictions[i])
        y_dict[pdb_ids[i]].append(y[i])

    with open(f"../results/{direction}{mode}{mode_}_dict.pkl", "wb") as f:
        pickle.dump((prediction_dict, y_dict), f)


    all_aucs = []
    all_auprs = []
    for pdb_id in prediction_dict:
        all_aucs.append(roc_auc_score(y_dict[pdb_id],
                                      prediction_dict[pdb_id]))
        all_auprs.append(average_precision_score(y_dict[pdb_id],
                                                 prediction_dict[pdb_id]))

    auc = np.mean(all_aucs)
    aupr = np.mean(all_auprs)

    all_laucs = []
    for pdb_id in prediction_dict:
        y = y_dict[pdb_id]
        y = np.array(y)
        predicted = prediction_dict[pdb_id]
        predicted = np.array(predicted)

        precisions, recalls, _ = \
            precision_recall_curve(y, predicted)

        sorted_indices = np.argsort(predicted)[::-1]
        y = y[sorted_indices]
        predicted = predicted[sorted_indices]
        num_pos = np.sum(y)

        all_laucs.append(get_log_auc(predicted, y, num_pos))

    log_auc = np.mean(all_laucs)

    results = [str(auc), str(aupr), str(log_auc)]

    for prec_val in [0.01, 0.05, 0.1, 0.25, 0.5]:
        all_recs = []
        for pdb_id in prediction_dict:
            y = y_dict[pdb_id]
            y = np.array(y)
            predicted = prediction_dict[pdb_id]
            predicted = np.array(predicted)

            precisions, recalls, _ = \
                precision_recall_curve(y, predicted)

            all_recs.append(get_recall(precisions, recalls, prec_val))

        results.append(str(np.mean(all_recs)))

    for prec_val in [0.01, 0.05, 0.1, 0.25, 0.5]:
        all_efs = []
        for pdb_id in prediction_dict:
            y = y_dict[pdb_id]
            y = np.array(y)
            predicted = prediction_dict[pdb_id]
            predicted = np.array(predicted)

            precisions, recalls, _ = \
                precision_recall_curve(y, predicted)

            sorted_indices = np.argsort(predicted)[::-1]
            y = y[sorted_indices]
            predicted = predicted[sorted_indices]
            num_pos = np.sum(y)

            all_efs.append(get_ef(y, prec_val, num_pos))

        results.append(str(np.mean(all_efs)))

    return f"{direction},{mode},{mode_},{','.join(results)}"


dirs = [dir_.replace("_dir_dict.pkl", "") for dir_
        in os.listdir("../../get_data/NewData/results/directions/")]
dirs = ["btd", "dtb"]

rows = [",".join(["direction","mode","mode_",
                  "AUC", "AUPR", "LogAUC", "recall_1", "recall_5",
                  "recall_10", "recall_25", "recall_50", "EF_1",
                  "EF_5", "EF_10", "EF_25", "EF_50"])]
from itertools import product

modes_ = ("_ifp", "_orig")
modes = ("_normal", "_wp", "_sp")
with PPE() as executor:
    rows_ = executor.map(get_performance, product(dirs, modes, modes_))
rows += list(rows_)
'''

for direction in dirs:
    for mode_ in ("_normal", "_sp", "_wp"):
        for mode in modes_:
            rows.append(get_performance((direction, mode_, mode)))
'''
'''


for direction in progressbar(dirs):
    row = get_performance(direction)
    rows.append(row)

'''

with open("../results/external_val.csv", "w") as f:
    f.write("\n".join(rows))

