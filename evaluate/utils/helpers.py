import numpy as np

def get_num_row_cols(n_plots):
    n_rows = int(np.ceil(np.sqrt(n_plots)))
    n_cols = int(np.ceil(n_plots / n_rows))

    return n_rows, n_cols


def write_metrics_to_file(save_file, metrics, class_border):
    with open(save_file, "w") as file:
        file.write("Class Border: {}\n".format(class_border))
        for metric, values in metrics.items():
            file.write("\n" + metric + "\n")
            line = "Average: {}".format(np.mean(values))
            line += "\tMin: {} (Step {})".format(np.min(values[1:]), np.argmin(values[1:]))
            line += "\tMax: {} (Step {})\n".format(np.max(values[1:]), np.argmax(values[1:]))
            file.write(line)
