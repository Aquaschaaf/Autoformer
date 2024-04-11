import matplotlib.pyplot as plt
import numpy as np

from evaluate.utils.helpers import get_num_row_cols

def plot_metrics_per_step(metrics, save_name=None):

    num_metrics = len(metrics)
    metric_names = list(metrics.keys())

    n_rows, n_cols = get_num_row_cols(num_metrics)
    fig, axs = plt.subplots(n_rows, n_cols)
    axs = axs.reshape(-1,)

    for i, metric in enumerate(metric_names):
        axs[i].plot(metrics[metric])
        axs[i].set_title(metric)
        # axs[i].set_ylim([0, 1])
        axs[i].set_xlabel("Prediction Step")

    if save_name is not None:
        plt.savefig(save_name)


def plot_correlation_pred_true_pct(data):

    num_steps = data["pred"].shape[0]
    correlation_per_step = []
    for i in range(num_steps):
        correlation_per_step.append(np.corrcoef(data["pred"][i, :], data["true"][i, :])[0, 1])

    plt.figure()
    plt.plot(correlation_per_step)
    plt.title("PCC correlation [-1, 1]")
    plt.xlabel("Prediction Step")
    plt.ylabel("Correlation Percentage Change Prediction vs. True")

    max_correlation = np.argmax(correlation_per_step)
    tmp = data["pred"][max_correlation, :]
    arr1inds = data["pred"][max_correlation, :].argsort()
    sorted_arr1 = data["pred"][max_correlation, :][arr1inds]
    sorted_arr2 = data["true"][max_correlation, :][arr1inds]

    plt.figure()
    x = list(range(len(sorted_arr1)))
    plt.scatter(x, sorted_arr1, label="(Sorted) Predictions")
    plt.scatter(x, sorted_arr2, label="True Values")
    plt.title("Predicted and True Pct Change for step {} (max correlation)".format(max_correlation))
    plt.ylabel("Value")






