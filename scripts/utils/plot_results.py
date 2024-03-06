import matplotlib.pyplot as plt
import numpy as np
import os

RESULT_DIR = "./results"
EXPERIMENT = "Exchange_96_96_Autoformer_custom_ftS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0"

seq_length = int(EXPERIMENT.split("_sl")[-1].split("_")[0])
label_length = int(EXPERIMENT.split("_ll")[-1].split("_")[0])
pred_length = int(EXPERIMENT.split("_pl")[-1].split("_")[0])

experiment_dir = os.path.join(RESULT_DIR, EXPERIMENT)
metrics_file = os.path.join(experiment_dir, "metrics.npy")
pred_file = os.path.join(experiment_dir, "pred.npy")
true_file = os.path.join(experiment_dir, "true.npy")
real_preds_file = os.path.join(experiment_dir, "real_prediction.npy")

metrics = np.load(metrics_file)
pred = np.load(pred_file)
true = np.load(true_file)
real_preds = np.load(real_preds_file) if os.path.isfile(real_preds_file) else None

def plot_prediction_and_gt(pred, gt, seq_length, plot_features=None):

    assert pred.shape == gt.shape, "Shape of predictions and ground truth mismatch! This is unexpected"

    num_sequences = gt.shape[0]
    if plot_features is not None:
        num_plots = len(plot_features)
    else:
        num_plots = pred.shape[-1]
    n_rows = int(np.ceil(np.sqrt(num_plots)))
    n_cols = int(np.ceil(num_plots / n_rows))

    fig, axs = plt.subplots(n_rows, n_cols)
    axs = np.reshape(axs, (-1,))

    for idx in range(num_plots):
        for seq_idx in range(0, num_sequences, seq_length+1):
            x = range(seq_idx, seq_idx+seq_length)
            p, = axs[idx].plot(x, pred[seq_idx, :, idx], 'g', label="Prediction")
            g, = axs[idx].plot(x, gt[seq_idx, :, idx], 'r', label="Ground Truth")

    fig.subplots_adjust(bottom=0.2, wspace=0.33)
    axs[0].set_title("Prediction and GT")
    axs[-1].legend(handles=[p, g], labels=['Predictions', 'Ground Truth'], loc='upper center',
                 bbox_to_anchor=(0.1, -0.3), fancybox=False, shadow=False, ncol=3)
    plt.show()

plot_prediction_and_gt(pred, true, seq_length)

plt.show()
