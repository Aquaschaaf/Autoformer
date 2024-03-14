import matplotlib.pyplot as plt
import numpy as np
import os

from evaluate.utils.data_loading import get_settings_from_experiment_name, load_data
from evaluate.utils.helpers import get_num_row_cols


def plot_prediction_and_gt(pred, gt, pred_length, plot_features=None):

    assert pred.shape == gt.shape, "Shape of predictions and ground truth mismatch! This is unexpected"

    num_sequences = gt.shape[0]
    if plot_features is not None:
        num_plots = len(plot_features)
    else:
        num_plots = pred.shape[-1]

    n_rows, n_cols = get_num_row_cols(num_plots)

    fig, axs = plt.subplots(n_rows, n_cols)
    axs = np.reshape(axs, (-1,))

    for idx in range(num_plots):
        for seq_idx in range(0, num_sequences, pred_length+1):
            x = range(seq_idx, seq_idx+pred_length)
            p, = axs[idx].plot(x, pred[seq_idx, :, idx], 'g', label="Prediction")
            g, = axs[idx].plot(x, gt[seq_idx, :, idx], 'r', label="Ground Truth")

    fig.subplots_adjust(bottom=0.2, wspace=0.33)
    axs[0].set_title("Prediction and GT")
    axs[-1].legend(handles=[p, g], labels=['Predictions', 'Ground Truth'], loc='upper center',
                 bbox_to_anchor=(0.1, -0.3), fancybox=False, shadow=False, ncol=3)


if __name__ == "__main__":
    RESULT_DIR = "../../results"
    EXPERIMENT = "Forex_96_96_Autoformer_custom_ftMS_sl60_ll30_pl30_dm32_nh8_el2_dl2_df32_fc3_ebtimeF_dtTrue_Exp_0_MCSAMPLING"
    experiment_dir = os.path.join(RESULT_DIR, EXPERIMENT)

    seq_length, pred_length = get_settings_from_experiment_name(EXPERIMENT)
    data = load_data(experiment_dir)
    plot_prediction_and_gt(data["pred"], data["true"], pred_length)
    plt.show()
