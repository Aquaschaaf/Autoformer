import torch
from argparse import ArgumentParser
import json
import os
import numpy as np
import pandas as pd
from utils.timefeatures import time_features
import argparse
import vectorbt as vbt
import matplotlib.pyplot as plt

from run import parse_arguments, build_settings_str_from_args
from exp.exp_inference import Exp_Inference
from models import Informer, Autoformer, Transformer, Reformer
from data_provider.data_factory import data_provider


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

    best_total_return = -10000
    best_pf = None
    best_threshhods = {"buy": None, "sell": None, "stop_loss": None}
    num_thresh_candidates = 5
    for i in range(num_thresh_candidates):
        for j in range(num_thresh_candidates):
            for k in range(num_thresh_candidates):

                buy_thresh = i/100.
                sell_thresh = -j/100.
                stop_loss = (k+3)/100.

                entries, exits = create_signals(df['predicted_pct_diff'], buy_thresh=buy_thresh, sell_thresh=sell_thresh)
                pf = vbt.Portfolio.from_signals(df["Close"], entries=entries, exits=exits, sl_stop=stop_loss, slippage=0.0025)  #   , slippage=0.0025, fixed_fees=0.5,
                # tr = pf.stats()["Win Rate [%]"]  #
                tr= pf.total_return()
                print("B {}, S {}, StopLoss {}: {}".format(buy_thresh, sell_thresh, stop_loss, tr))

                if tr > best_total_return:
                    best_pf = pf
                    best_total_return = tr
                    best_threshhods["buy"] = buy_thresh
                    best_threshhods["sell"] = sell_thresh
                    best_threshhods["sell"] = sell_thresh
                    best_threshhods["stop_loss"] = stop_loss

    print("Best Thresholds: {}".format(best_threshhods))
    stats = best_pf.stats()
    print(stats)
    best_pf.plot().show()


if __name__ == "__main__":

    CREATE_PREDICTION_DATA = False
    OUT_DIR = "results"
    MODEL_DIR = "D:\Mine\Autoformer\checkpoints\Exchange_96_96_Informer_ohlcv_ftS_sl40_ll20_pl10_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_'FixPctContinuity'_0"
    # DATA_FILE = r"D:\Mine\DATA\data\dax\BAYN.DE_hourly.csv"
    # DATA_FILE = r"D:\Mine\DATA\data\dax\BMW.DE_hourly.csv"
    # DATA_FILE = r"D:\Mine\DATA\data\crypto\ETH-USD_hourly.csv"
    DATA_FILE = r"D:\Mine\DATA\data\crypto\BTC-USD_hourly.csv"

    model_name = MODEL_DIR.split(os.sep)[-1]
    data_name = DATA_FILE.split(os.sep)[-1].split(".")[0]
    out_file = os.path.join(OUT_DIR, model_name, "{}_predictions.csv".format(data_name))

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
                "predicted_pct_diff": predicted_diff_pct[0],
                "true_diff_pct": true_diff_pct[0]
            }

        df = pd.DataFrame(results).transpose()
        df.to_csv(out_file, index_label='date')
        df["date"] = df.index

    else:
        df = pd.read_csv(out_file)

    df.date = pd.to_datetime(df.date).dt.tz_localize(None)
    data = data.merge(df, how="outer")
    data = data[data['predicted_pct_diff'].notnull()]

    backtest(data)