import os
import pandas as pd
import matplotlib.pyplot as plt

from evaluate.utils.data_loading import get_settings_from_experiment_name, load_data
from evaluate.utils.eval import eval_per_predicted_step, create_classification_metrics_per_step, create_error_metrics_per_step
from evaluate.utils.helpers import write_metrics_to_file
from evaluate.utils.backtest import simple_backtest, backtest
from evaluate.plot.plot_metrics import plot_metrics_per_step, plot_correlation_pred_true_pct



EXPERIMENT_DIR = "/home/matthias/Projects/Autoformer/results/BTC_Informer_ohlcv_ftS_sl60_ll30_pl30_dm512_nh8_el2_dl2_df2048_fc3_ebtimeF_dtTrue_'Exp'_0"
# EXPERIMENT_DIR = "/home/matthias/Projects/Autoformer/results/BTC_Autoformer_ohlcv_ftMS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_'Exp'_0"
CLASS_BORDER = {"short": -0.01, "long": 0.01}
EVAL_DIR = "eval_results"
SAMPLE_LIMIT = None
RAW_DATA_FILE = "/home/matthias/Projects/Autoformer/dataset/btc_usd/BTC-Hourly.csv"
raw_data = pd.read_csv(RAW_DATA_FILE)
exp_name = EXPERIMENT_DIR.split(os.sep)[-1]
results_dir = os.path.join(EVAL_DIR, exp_name)
metrics_file = os.path.join(results_dir, "{}.txt".format(exp_name))

if not os.path.exists(EVAL_DIR):
    os.mkdir(EVAL_DIR)
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

seq_length, pred_length = get_settings_from_experiment_name(exp_name)
pred_data = load_data(EXPERIMENT_DIR)

# Apply SAmple limit
if SAMPLE_LIMIT is not None:
    for data_type, values in pred_data.items():
        if values is None:
            continue
        else:
            pred_data[data_type] = values[-SAMPLE_LIMIT:]

# Plot results

# Evaluate classification
pred_classes, true_classes, pct_errors, pct_per_step = eval_per_predicted_step(pred_data, CLASS_BORDER)

c_metrics = create_classification_metrics_per_step(pred_classes, true_classes)
e_metrics = create_error_metrics_per_step(pct_errors)
write_metrics_to_file(metrics_file, {**c_metrics, **e_metrics}, CLASS_BORDER)
plot_metrics_per_step(c_metrics, save_name=os.path.join(results_dir, "classification_metrics.png"))
plot_metrics_per_step(e_metrics, save_name=os.path.join(results_dir, "error_metrics.png"))

plot_correlation_pred_true_pct(pct_per_step)

backtest(raw_data["Close"][-pred_classes.shape[0]:], pred_classes, fees=0.00, slippage=0.00, use_step=-1)
# simple_backtest(pred_data["final_input"], pred_classes, use_step=10)
# Create the Signals Portfolio


plt.show()

# if __name__ == "__main__":
#     # run_evaluation()
#     pass


