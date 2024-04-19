import torch
from argparse import ArgumentParser
import json
import os
import numpy as np
import pandas as pd
from plotly.offline import plot
import subprocess

from utils.timefeatures import time_features
import argparse
import itertools
import vectorbt as vbt
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from run import parse_arguments, build_settings_str_from_args
from exp.exp_inference import Exp_Inference
from models import Informer, Autoformer, Transformer, Reformer
from data_provider.data_factory import data_provider

PREDICTION_COL = "Prediction"
GT_COL = "GT"

GT_SELL_THRESH = -0.01
GT_BUY_THRESH = 0.01

FEES = 0.0
SLIPPAGE = 0.0

def args_inference():
    N_MODEL_ITERATIONS = 1
    SAMPLE_LIMIT = 25000  # None

    args = parse_arguments()
    # Add Sampling arguments
    args_dict = vars(args)
    args_dict["mc_dropout"] = True
    args_dict["mc_samples"] = 50

    out_dir_suffix = "_mcsampling{}".format(args_dict["mc_samples"]) if args_dict["mc_dropout"] else None

    for ii in range(N_MODEL_ITERATIONS):
        settings = build_settings_str_from_args(args, ii)

        exp = Exp_Inference(args)  # set experiments
        exp.test(settings, sample_limit=SAMPLE_LIMIT, test=1, out_dir_suffix=out_dir_suffix)
        torch.cuda.empty_cache()


class PredictionModel:
    def __init__(self, exp_dir):
        self.args = self.load_config(exp_dir)
        self.model = self.load_model(exp_dir).to("cuda:0")
        pass

    def load_config(self, config_dir):
        parser = ArgumentParser()
        args = parser.parse_args()
        with open(os.path.join(config_dir, 'config.txt'), 'r') as f:
            args.__dict__ = json.load(f)
        return args

    def load_model(self, exp_dir):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
        }

        model = model_dict[self.args.model].Model(self.args).float()
        model.load_state_dict(torch.load(os.path.join(exp_dir, 'checkpoint.pth')))
        return model


def calc_percent(start, end):
    return (end - start) / start

def plot_input_output(input, prediction, label, datetime, label_len, pred_len):

    data_freq = pd.infer_freq(datetime)
    start = datetime.iloc[-1]
    prediction_times = pd.date_range(start=start, periods=pred_len + 1, freq=data_freq)
    datetime = pd.DataFrame({'date': prediction_times}).merge(datetime, how='outer')

    label_start = len(input)-label_len

    plt.figure()
    plt.plot(datetime.iloc[:len(input)], input, label="Input")
    plt.plot(datetime.iloc[label_start:label_start+len(label)], label, '--', label="Label")
    plt.plot(datetime.iloc[-len(prediction):], prediction, label="Prediction")
    plt.legend()
    plt.show()


def backtest(df):

    def create_signals(predictions, buy_thresh=0.02, sell_thresh=-0.05):
        entries = predictions > buy_thresh
        exits = predictions < sell_thresh
        return entries, exits

    def set_false_in_series(series, indices, num_false):
        series = series.astype(bool)
        max_index = max(entries.index.tolist())
        for index in indices:
            end_index = min(index + num_false, max_index)
            series.loc[index:end_index] = False
        return series

    best_total_return = -10000
    best_pf = None
    best_thresholds = {"buy": None, "sell": None, "stop_loss": None}
    num_thresh_candidates = 10
    for i in range(1, num_thresh_candidates):
        for j in range(1, num_thresh_candidates):
            for k in range(3, num_thresh_candidates):
                for l in range(5, 30, 5):

                    buy_thresh = i/100.
                    sell_thresh = -j/100.
                    stop_loss = k/100.
                    sl_cooldown = l

                    entries, exits = create_signals(df[PREDICTION_COL], buy_thresh=buy_thresh, sell_thresh=sell_thresh)
                    pf = vbt.Portfolio.from_signals(df["Close"], entries=entries, exits=exits, sl_stop=stop_loss, slippage=SLIPPAGE, fixed_fees=FEES)  #   , slippage=0.0025, fixed_fees=0.5,

                    # ==========================================================================================
                    # Identify days where stop loss was triggered
                    stop_orders_ts = pf.exit_trades.records_readable['Exit Timestamp'].tolist()
                    # Ensure the list contains valid indices
                    entries = set_false_in_series(entries, stop_orders_ts, num_false=sl_cooldown)
                    pf = vbt.Portfolio.from_signals(df["Close"], entries=entries, exits=exits,sl_stop=stop_loss, slippage=SLIPPAGE, fixed_fees=FEES)
                    # ==========================================================================================

                    # tr = pf.stats()["Win Rate [%]"]  #
                    tr = pf.total_return()
                    print("B {}, S {}, StopLoss {}, Cooldwon{}: {}".format(buy_thresh, sell_thresh, stop_loss, tr, sl_cooldown))

                    if tr > best_total_return:
                        best_pf = pf
                        best_total_return = tr
                        best_thresholds["buy"] = buy_thresh
                        best_thresholds["sell"] = sell_thresh
                        best_thresholds["stop_loss"] = stop_loss
                        best_thresholds["sl_cooldown"] = sl_cooldown

    print("Best Thresholds: {}".format(best_thresholds))
    stats = best_pf.stats()
    print(stats)

    best_pf.plot().show()

    return best_pf, best_thresholds


def find_best_class_threshold(predictions, true_values, thresholds):

    use_precision = False

    best_metric_value = 0
    best_threshold = None
    # Find best value for high threshold
    for thresh_vals in thresholds:

        pred_classes = assign_classes(predictions, thresh_vals["sell"], thresh_vals["buy"])
        # true_classes = assign_classes(true_values, GT_SELL_THRESH, GT_BUY_THRESH)
        true_classes = assign_classes(true_values, thresh_vals["sell"], thresh_vals["buy"])

        # Calculate metrics
        if use_precision:
            precision, recall, f1, _ = precision_recall_fscore_support(true_classes, pred_classes, average=None,
                                                                       labels=[-1, 0, 1])
            metric = precision[0] + precision[-1]
        else:
            conf_matrix = confusion_matrix(true_classes, pred_classes, labels=[-1, 0, 1])
            metric = conf_matrix[-1, -1] + conf_matrix[0, 0] - (conf_matrix[0, -1] / 1) #+ conf_matrix[1, 1] #

        if metric > best_metric_value:
            best_metric_value = metric
            best_threshold = thresh_vals

        #print("Threshold values: {}. Metric: {}".format(thresh_vals, metric))

    print("Best Threshold: {}. Metric: {}".format(best_threshold, best_metric_value))
    return best_threshold


def generate_classification_metrics(df, best_thresholds=None):

    if best_thresholds is None:
        param_grid = {
            "sell": [-x / 100.0 for x in range(1, 11)],
            "buy": [x / 100.0 for x in range(1, 11)],
        }
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        best_thresholds = find_best_class_threshold(df[PREDICTION_COL], df[GT_COL], param_combinations)

    accuracy, precision, recall, f1, conf_matrix = evaluate_classification(df[PREDICTION_COL],
                                                                           df[GT_COL],
                                                                           best_thresholds["sell"],
                                                                           best_thresholds["buy"])

    print("Precision with both best thresholds {}: {}".format(best_thresholds, precision))
    plot_confusion_matrix(conf_matrix, classes=[-1, 0, 1])

    metrics = {
        "best_threshold": best_thresholds,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "conf_matrix": conf_matrix
    }
    return metrics

def assign_classes(values, threshold_low, threshold_high):
    """Assign classes based on thresholds."""
    return np.select(
        [values < threshold_low, values <= threshold_high, values > threshold_high],
        [-1, 0, 1]
    )


def evaluate_classification(predictions, true_values, threshold_low, threshold_high):
    """Evaluate the classification using various metrics."""
    pred_classes = assign_classes(predictions, threshold_low, threshold_high)
    # true_classes = assign_classes(true_values, GT_SELL_THRESH, GT_BUY_THRESH)
    true_classes = assign_classes(true_values, threshold_low, threshold_high)

    # Calculate metrics
    accuracy = accuracy_score(true_classes, pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(true_classes, pred_classes, average=None,
                                                               labels=[-1, 0, 1])
    conf_matrix = confusion_matrix(true_classes, pred_classes, labels=[-1, 0, 1])

    return accuracy, precision, recall, f1, conf_matrix


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """This function prints and plots the confusion matrix."""
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 3.
    for i, j in enumerate(classes):
        for k, class_label in enumerate(classes):
            plt.text(i, k, f"{cm_norm[k][i]:.2f}\n{cm[k][i]}",
                     horizontalalignment="center",
                     color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def generate_metrics(df, best_thresholds=None):


    df = df[df[GT_COL].notnull()]

    # Calculate evaluation metrics
    mae = mean_absolute_error(df[GT_COL], df[PREDICTION_COL])
    mse = mean_squared_error(df[GT_COL], df[PREDICTION_COL])
    rmse = np.sqrt(mse)
    r_squared = r2_score(df[GT_COL], df[PREDICTION_COL])
    classi_metrics = generate_classification_metrics(df, best_thresholds)

    # Print the results
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared:", r_squared)

    all_metrics = {}
    all_metrics["regression"] = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r_squared": r_squared
    }
    all_metrics["classification"] = classi_metrics

    return all_metrics



def summarize(best_thresholds, pf, metrics, out_dir, meta_info):

    def format_conf_mat(mat, suffix=None):
        # Creating a format string: adjust precision and width as needed
        # This format string sets up spaces for alignment: right-aligned by default
        if mat.shape != (3, 3):
            raise ValueError("Input must be a 3x3 matrix.")
        formatter = "{:>10.4f} {:>10.4f} {:>10.4f}\n"
        formatted_string = "Confusion Matrix\n" if suffix is None else "Confusion Matrix {}\n".format(suffix)
        for row in mat:
            # Append each formatted row to the formatted_string
            formatted_string += formatter.format(*row)
        return formatted_string

    def create_section_header(name):
        out_str = ''
        sep_line = "\n===========================================\n"
        out_str += sep_line
        out_str += name.upper()
        out_str += sep_line
        return out_str

    def get_git_revision_short_hash() -> str:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

    out_file = os.path.join(out_dir, "evaluation_summary.txt")
    plot(pf.plot(), filename=os.path.join(out_dir, 'portfolio.html'))

    with open(out_file, "w") as file:
        file.write(create_section_header("META INFORMATION"))
        for info_name, info in meta_info.items():
            file.write("{}: {}\n".format(info_name, info))
        file.write("Git commit: {}\n".format(get_git_revision_short_hash()))
        file.write(create_section_header("used thresholds"))
        file.write("{}\n".format(best_thresholds))
        for task, task_metrics in metrics.items():
            file.write(create_section_header(task))
            for metric_name, metric_val in task_metrics.items():
                if metric_name == "conf_matrix":
                    file.write(format_conf_mat(metric_val))
                    conf_mat_normed = metric_val.astype('float') / metric_val.sum(axis=1)[:, np.newaxis]
                    file.write(format_conf_mat(conf_mat_normed, suffix="Normed"))
                else:
                    file.write("{}: {}\n".format(metric_name, metric_val))
        file.write("\n")
        file.write(create_section_header("PORTFOLIO STATS"))
        file.write(pf.stats().to_string())


if __name__ == "__main__":

    CREATE_PREDICTION_DATA = False
    BASE_OUT_DIR = "results"
    MODEL_DIR = "/home/matthias/Projects/Autoformer/checkpoints/BTC_Informer_ohlcv_ftS_sl60_ll20_pl10_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_'des'_0"
    # DATA_FILE = r"D:\Mine\DATA\data\dax\BAYN.DE_hourly.csv"
    # DATA_FILE = r"D:\Mine\DATA\data\dax\BMW.DE_hourly.csv"
    # DATA_FILE = r"D:\Mine\DATA\data\crypto\ETH-USD_hourly.csv"
    DATA_FILE = r"/home/matthias/Projects/Autoformer/dataset/btc_usd/BTC-Hourly.csv"

    model_name = MODEL_DIR.split(os.sep)[-1]
    data_name = DATA_FILE.split(os.sep)[-1].split(".")[0]
    out_dir = os.path.join(BASE_OUT_DIR, model_name)
    out_file = os.path.join(out_dir, "{}_predictions.csv".format(data_name))
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    data = pd.read_csv(DATA_FILE)
    data["date"] = pd.to_datetime(data["date"], errors='coerce')
    data.date = data.date.dt.tz_localize(None)
    df_raw = data.sort_values(by='date')

    if not os.path.isfile(out_file) and not CREATE_PREDICTION_DATA:
        print("Data creation is set to False but outfile does not exist. Generating new one")
        CREATE_PREDICTION_DATA = True

    if CREATE_PREDICTION_DATA:
        predictor = PredictionModel(MODEL_DIR)
        predictor.args.data_path = os.path.basename(DATA_FILE)
        predictor.args.is_training = False

        seq_length = predictor.args.seq_len
        label_length = predictor.args.label_len
        pred_length = predictor.args.pred_len
        target_col = predictor.args.target
        seq_label_len = seq_length + label_length

        # Set the dataset to the one specified above
        predictor.args.data_path = os.path.basename(DATA_FILE)
        predictor.args.root_path = os.path.dirname(DATA_FILE)

        # TO COMPARE INPUT
        data_set, data_loader = data_provider(predictor.args, "test")
        max_range = len(data_set.data_x) - seq_length + 1  # +1 to include final prediction
        results = {}
        for i in range(0, max_range):
            seq_x, seq_y, seq_x_mark, seq_y_mark, datetime = data_set.__getitem__(i)

            # Add batch dimension
            _seq_x = torch.Tensor(np.expand_dims(seq_x, axis=0)).to("cuda:0")
            _seq_y = torch.Tensor(np.expand_dims(seq_y, axis=0)).to("cuda:0")
            _seq_x_mark = torch.Tensor(np.expand_dims(seq_x_mark, axis=0)).to("cuda:0")
            _seq_y_mark = torch.Tensor(np.expand_dims(seq_y_mark, axis=0)).to("cuda:0")
            # Prepare decoder input
            dec_inp = torch.zeros_like(_seq_y[:, :pred_length, :]).float()
            dec_inp = torch.cat([_seq_y[:, :label_length, :], dec_inp], dim=1).float().to("cuda:0")

            # Do prediction
            outputs = predictor.model(_seq_x, _seq_x_mark, dec_inp, _seq_y_mark)
            outputs = np.squeeze(outputs.detach().cpu().numpy())

            # Debug plot
            # if i >= 0:
            #     x = np.squeeze(seq_x)
            #     label = np.squeeze(seq_y)
            #     plot_input_output(x, outputs, label, datetime, label_length, pred_length)

            # Track metrics if ground trtuth label are available
            predicted_diff_pct = outputs[-1] - seq_x[-1]
            true_diff_pct = seq_y[-1] - seq_x[-1] if i < max_range - pred_length else [None]
            results[datetime.iloc[-1]] = {
                PREDICTION_COL: predicted_diff_pct[0],
                GT_COL: true_diff_pct[0]
            }

        df = pd.DataFrame(results).transpose()
        df.to_csv(out_file, index_label='date')
        df["date"] = df.index

    else:
        df = pd.read_csv(out_file)

    df.date = pd.to_datetime(df.date).dt.tz_localize(None)
    data = data.merge(df, how="outer")
    data = data[data[PREDICTION_COL].notnull()]

    pf, best_thresholds = backtest(data)
    metrics = generate_metrics(data, best_thresholds)

    meta_info = {"Model": MODEL_DIR, "Dataset": DATA_FILE, "Fees": FEES, "Slippage": SLIPPAGE}
    summarize(best_thresholds, pf, metrics, out_dir, meta_info)

    plt.show()

