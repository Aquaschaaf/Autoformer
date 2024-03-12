import matplotlib.pyplot as plt

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

