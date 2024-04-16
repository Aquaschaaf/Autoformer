import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, accuracy_score, precision_score, recall_score
from evaluate.utils.helpers import write_metrics_to_file

def eval_per_predicted_step(data, class_border, target_idx=-1):

    def _pct_change(current, previous):
        if current == previous:
            return 1.0
        try:
            return (current - previous) / previous
        except ZeroDivisionError:
            return 0.

    def assign_classes(data, class_border):

        class_labels = np.empty_like(data)

        msk = data < class_border["short"]
        class_labels[msk] = 0

        msk = np.ma.masked_inside(data, class_border["short"], class_border["long"]).mask
        class_labels[msk] = 1

        msk = (data >= class_border["long"])
        class_labels[msk] = 2

        return class_labels.astype(np.int32)


    true = data["true"]
    pred = data["pred"]
    final_input = data["final_input"]

    num_sequences = true.shape[0]

    pred_classes, true_classes, errors = [], [], []
    pct_change_per_step = {"true": [], "pred": []}
    for seq_idx in range(num_sequences):

        fi = final_input[seq_idx]
        pct_pred = pred[seq_idx, :, target_idx]  # np.array([_pct_change(xi, fi) for xi in pred[seq_idx, :, target_idx]])
        pct_true = true[seq_idx, :, target_idx]  # np.array([_pct_change(xi, fi) for xi in true[seq_idx, :, target_idx]])

        correlation = np.corrcoef(pct_pred, pct_true)

        pred_classes.append(assign_classes(pct_pred, class_border))
        # pred_classes.append((pct_pred >= class_border).astype(np.int32))
        true_classes.append(assign_classes(pct_true, class_border))
        # true_classes.append((pct_true >= class_border).astype(np.int32))
        errors.append(pct_pred - pct_true)

        pct_change_per_step["true"].append(pct_true)
        pct_change_per_step["pred"].append(pct_pred)

        # Debug Plot
        # fig, (ax_raw, ax_pct, ax_class) = plt.subplots(3,1)
        # ax_raw.plot(true[seq_idx, :, target_idx], label="TRUE")
        # ax_raw.plot(pred[seq_idx, :, target_idx], label="Pred")
        # ax_raw.legend()
        # ax_pct.plot(pct_true, label="PCT True")
        # ax_pct.plot(pct_pred, label="PCT Pred")
        # ax_pct.plot(pct_pred - pct_true, label="Diff")
        # ax_pct.legend()
        # ax_class.plot((pct_true >= class_border).astype(np.int32), label="CLASS True")
        # ax_class.plot((pct_pred >= class_border).astype(np.int32), label="CLASS Pred")
        # ax_class.legend()
        # plt.show()

    pct_change_per_step["true"] = np.array(pct_change_per_step["true"]).transpose()
    pct_change_per_step["pred"] = np.array(pct_change_per_step["pred"]).transpose()

    return np.array(pred_classes), np.array(true_classes), np.array(errors), pct_change_per_step

def create_classification_metrics_per_step(pred_classes, true_classes, save_file=None):

    metrics = {}
    n_steps = true_classes.shape[-1]
    for step_idx in range(n_steps):

        step_metrics = {}
        pred = pred_classes[:, step_idx].reshape((-1, 1))
        true = true_classes[:, step_idx].reshape((-1, 1))

        step_metrics["average_precision"] = average_precision_score(true, pred)
        step_metrics["accuracy"] = accuracy_score(true, pred)
        step_metrics["precision"] = precision_score(true, pred, average='micro')
        step_metrics["recall"] = recall_score(true, pred, average='micro')

        metrics["{}".format(step_idx)] = step_metrics

    # Resort metrics
    metric_names = list(metrics[list(metrics.keys())[0]].keys())
    for metric in metric_names:
        all_values = []
        for step_idx in range(n_steps):
            all_values.append(metrics[str(step_idx)][metric])
        metrics[metric] = all_values
    metrics = {n: m for n, m in metrics.items() if n in metric_names}

    return metrics

def create_error_metrics_per_step(errors, save_file=None):

    metrics = {}
    n_steps = errors.shape[-1]
    for step_idx in range(n_steps):

        step_metrics = {}
        error = errors[:, step_idx]

        step_metrics["max_error"] = np.max(error)
        step_metrics["mean_abs_error"] = np.mean(abs(error))

        metrics["{}".format(step_idx)] = step_metrics

    # Resort metrics
    metric_names = list(metrics[list(metrics.keys())[0]].keys())
    for metric in metric_names:
        all_values = []
        for step_idx in range(n_steps):
            all_values.append(metrics[str(step_idx)][metric])
        metrics[metric] = all_values
    metrics = {n: m for n, m in metrics.items() if n in metric_names}

    return metrics





