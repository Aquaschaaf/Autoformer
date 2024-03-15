import os
import numpy as np
import pandas as pd

def get_settings_from_experiment_name(exp_name):
    seq_length = int(exp_name.split("_sl")[-1].split("_")[0])
    label_length = int(exp_name.split("_ll")[-1].split("_")[0])
    pred_length = int(exp_name.split("_pl")[-1].split("_")[0])

    return seq_length, pred_length

def load_data(exp_dir):
    metrics_file = os.path.join(exp_dir, "metrics.npy")
    pred_file = os.path.join(exp_dir, "pred.npy")
    true_file = os.path.join(exp_dir, "true.npy")
    real_preds_file = os.path.join(exp_dir, "real_prediction.npy")
    final_inputs_file = os.path.join(exp_dir, "final_inputs.npy")

    metrics = np.load(metrics_file) if os.path.isfile(metrics_file) else None
    pred = np.load(pred_file)
    true = np.load(true_file)
    final_inputs = np.load(final_inputs_file) if os.path.isfile(final_inputs_file) else None
    real_preds = np.load(real_preds_file) if os.path.isfile(real_preds_file) else None

    data = {
        "metrics": metrics,
        "pred": pred,
        "true": true,
        "final_input": final_inputs,
        "real_preds": real_preds
    }

    return data

def load_raw_data(file):

    return pd.read_csv(file)
