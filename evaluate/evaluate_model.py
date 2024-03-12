import os
import matplotlib.pyplot as plt

from evaluate.utils.data_loading import get_settings_from_experiment_name, load_data
from evaluate.utils.eval import eval_per_predicted_step, create_classification_metrics_per_step, create_error_metrics_per_step
from evaluate.utils.helpers import write_metrics_to_file
from evaluate.plot.plot_metrics import plot_metrics_per_step


EXPERIMENT_DIR = "/home/matthias/Projects/Autoformer/results/Exchange_96_96_Autoformer_custom_ftMS_sl60_ll30_pl30_dm32_nh8_el2_dl2_df32_fc3_ebtimeF_dtTrue_Exp_0"
CLASS_BORDER = 0.0
EVAL_DIR = "eval_results"
SAMPLE_LIMIT = None

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
pred_classes, true_classes, pct_errors = eval_per_predicted_step(pred_data, CLASS_BORDER)
c_metrics = create_classification_metrics_per_step(pred_classes, true_classes)
e_metrics = create_error_metrics_per_step(pct_errors)

write_metrics_to_file(metrics_file, {**c_metrics, **e_metrics}, CLASS_BORDER)

plot_metrics_per_step(c_metrics, save_name=os.path.join(results_dir, "classification_metrics.png"))
plot_metrics_per_step(e_metrics, save_name=os.path.join(results_dir, "error_metrics.png"))
plt.show()


