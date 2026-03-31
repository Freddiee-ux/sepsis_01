__author__ = 'Aaron J Masino'

import os
import sys
from glob import glob

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from sepsis import mathx
import sepsis.evaluation as evaluate
from sepsis import log_worker as slog
import numpy as np

# ************************** GLOBAL PARAMETERS ******************************************
data_dir = os.path.join(_project_root, "data")

# Default to the directory produced by `python -m sepsis.main_train_eval`.
# Override by setting `SEPSIS_PRED_PROB_DIR`.
input_data_dir = os.environ.get(
    "SEPSIS_PRED_PROB_DIR",
    os.path.join(data_dir, "interim", "prediction_probabilities"),
)

_prob_file = os.path.join(input_data_dir, "{0}_pred_probs.csv")
_targ_file = os.path.join(input_data_dir, "{0}_targets.csv")

target_metric_name = evaluate.SENSITIVITY
target_metric_value = 0.8
ci_level = 0.95

interim_dir = os.path.join(data_dir, "interim")
os.makedirs(interim_dir, exist_ok=True)

metrics_output_file = os.path.join(
    interim_dir,
    "scoring_metrics_fixed_{0}_{1}.csv".format(target_metric_name, target_metric_value),
)
metrics_ranges_output_file = os.path.join(
    interim_dir,
    "scoring_metrics_ranges_fixed_{0}_{1}.csv".format(target_metric_name, target_metric_value),
)
# ************************* GLOBAL PARAMETERS END *************************************


def loaddata(file):
    with open(file, "r") as f:
        all_data = []
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            all_data.append([float(x) for x in line.split(",") if x != ""])
        return all_data


if not os.path.isdir(input_data_dir):
    raise FileNotFoundError(
        f"Prediction probability directory not found: {input_data_dir}. "
        "Set SEPSIS_PRED_PROB_DIR or run `python -m sepsis.main_train_eval` first."
    )

# Auto-detect which model prefixes are available (avoid crashing on missing files).
prob_files = glob(os.path.join(input_data_dir, "*_pred_probs.csv"))
targ_files = glob(os.path.join(input_data_dir, "*_targets.csv"))
prob_prefixes = {os.path.basename(p)[: -len("_pred_probs.csv")] for p in prob_files}
targ_prefixes = {os.path.basename(t)[: -len("_targets.csv")] for t in targ_files}
file_prefixes = sorted(prob_prefixes & targ_prefixes)

if not file_prefixes:
    raise FileNotFoundError(
        f"No matching '*_pred_probs.csv'/'*_targets.csv' pairs found in: {input_data_dir}."
    )

# Avoid appending duplicate headers across multiple runs.
for out_file in (metrics_output_file, metrics_ranges_output_file):
    if os.path.exists(out_file):
        os.remove(out_file)

line = (
    "model,acc,acc_std,acc_cil,acc_cih,f1,f1_std,f1_cil,f1_cih,"
    "sensitivity,sensitivity_std,sensitivity_cil,sensitivity_cih,"
    "specificity,specificity_std,specificity_cil,specificity_cih,"
    "precision,precision_std,precision_cil,precision_cih,"
    "npv,npv_std,npv_cil,npv_cih\n"
)
slog.log_line(line, metrics_output_file)

range_line = (
    "model,acc,acc_low,acc_high,f1,f1_low,f1_high,"
    "sensitivity,sensitivity_low,sensitivity_high,"
    "specificity,specificity_low,specificity_high,"
    "precision,precision_low,precision_high,"
    "npv,npv_low,npv_high\n"
)
slog.log_line(range_line, metrics_ranges_output_file)

for fp in file_prefixes:
    probs = loaddata(_prob_file.format(fp))
    targs = loaddata(_targ_file.format(fp))
    acc, f1, sen, spec, precis, npv = evaluate.compute_metrics(
        targs, probs, target_metric_value, target_metric_name
    )
    scores = [acc, f1, sen, spec, precis, npv]

    line = "{0},".format(fp)
    range_line = "{0},".format(fp)
    for score in scores:
        m, s, cil, cih = mathx.mean_confidence_interval(score, ci_level)
        line = "{0}{1},{2},{3},{4},".format(line, m, s, cil, cih)
        low = np.min(score)
        high = np.max(score)
        range_line = "{0}{1},{2},{3},".format(range_line, m, low, high)

    slog.log_line("{0}\n".format(line[0:-1]), metrics_output_file)
    slog.log_line("{0}\n".format(range_line[0:-1]), metrics_ranges_output_file)

