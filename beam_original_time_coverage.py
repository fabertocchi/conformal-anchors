# ---------------------------------------------------------
# BEAM SIZE SENSITIVITY (ORIGINAL MODE)
# Coverage vs beam size + Runtime vs beam size
# - 20 test instances
# - for each instance: exclude ONE label in the SAME disease group (≠ true), chosen at random
# - original mode only
# - writes a full log to an output .txt file
# ---------------------------------------------------------

from anchor.anchor.anchor_tabular import AnchorTabularExplainer
from model_training import (
    X_train, y_train, X_conf_pred, y_conf_pred,
    X_anchors, y_anchors, X_test, y_test,
    X_train_orig, y_train_orig,
    X_conf_pred_orig, y_conf_pred_orig,
    X_anchors_orig, y_anchors_orig,
    X_test_orig, y_test_orig,
    categories,
)
import xgboost as xgb
from subset import get_knn_subsets
from prediction_set import compute_conformal_prediction_set_batch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

# Reproducibility
np.random.seed(1)
rng = np.random.default_rng(1)

# -----------------------------
# Settings
# -----------------------------
beam_sizes = [1, 3, 5, 7, 10, 12, 15]
n_instances = 50
k_neighbors = 100
alpha = 0.01

# Disease groups in your synthetic setup 
labels_stomach = [1, 5, 6, 7, 8]
labels_lung = [0, 2, 3, 4, 9]

generic_symptoms_cols = [
    'fever_severity', 'cough_severity', 'chest_pain_severity',
    'abdominal_pain_severity', 'fatigue_level', 'nausea'
]

# -----------------------------
# Load model
# -----------------------------
xgb_cl = xgb.XGBClassifier()
xgb_cl.load_model("xgb_model.json")
class_names = xgb_cl.classes_

# -----------------------------
# Helper: pick one label at random in same disease group (≠ true)
# -----------------------------
def pick_same_group_label_to_exclude_random(true_label, rng_local):
    if int(true_label) in labels_lung:
        group = labels_lung
    elif int(true_label) in labels_stomach:
        group = labels_stomach
    else:
        group = [int(c) for c in class_names]

    candidates = [int(l) for l in group if int(l) != int(true_label)]
    if len(candidates) == 0:
        return None
    return int(rng_local.choice(candidates))

# -----------------------------
# Storage: per-beam lists of coverages/runtimes (across instances)
# -----------------------------
coverage_by_beam = {b: [] for b in beam_sizes}
runtime_by_beam = {b: [] for b in beam_sizes}
skipped_instances = 0

# -----------------------------
# Select instances
# -----------------------------
n_instances = min(n_instances, len(X_test))
selected_test_indices = np.random.choice(len(X_test), size=n_instances, replace=False)

# -----------------------------
# Output file
# -----------------------------
output_path = "beam_size_sensitivity_original_mode.txt"

# -----------------------------
# Main experiment (logged to file)
# -----------------------------
with open(output_path, "w") as f:
    f.write("BEAM SIZE SENSITIVITY (ORIGINAL MODE)\n")
    f.write("Coverage vs beam size + Runtime vs beam size\n")
    f.write(f"Number of test instances: {n_instances}\n")
    f.write(f"k (neighbors): {k_neighbors}\n")
    f.write(f"alpha: {alpha}\n")
    f.write(f"Beam sizes: {beam_sizes}\n")
    f.write("Label-to-exclude: ONE random label in the same disease group (≠ true label)\n")
    f.write("=" * 100 + "\n\n")

    for run_id, test_idx in enumerate(selected_test_indices, 1):
        f.write("=" * 80 + "\n")
        f.write(f"Instance {run_id}/{n_instances} (test index = {test_idx})\n")

        # target patient (no masking)
        new_patient = X_test.iloc[test_idx].copy()
        true_label = int(y_test.iloc[test_idx])

        # choose label-to-exclude (random same group)
        label_to_exclude = pick_same_group_label_to_exclude_random(true_label, rng)

        if label_to_exclude is None:
            f.write("  -> No valid same-group label to exclude. Skipping instance.\n\n")
            skipped_instances += 1
            continue

        # Build 1-row DF for get_knn_subsets
        new_patient_df = new_patient.to_frame().T
        new_patient_df.index = [test_idx]

        # kNN neighborhood (binned + raw indices)
        subsets_list, y_subsets_list, similarities_list, indices_neighbors_list = get_knn_subsets(
            new_patient_df, X_anchors, y_anchors, k=k_neighbors
        )
        subset_new_patient = subsets_list[0]                 # binned neighbors
        y_subset_new_patient = y_subsets_list[0]             # neighbor labels
        indices_neighbors = indices_neighbors_list[0]        # indices into *_orig pools

        # conformal prediction sets on neighbors + qhat
        prediction_sets, qhat = compute_conformal_prediction_set_batch(
            xgb_cl, X_conf_pred, y_conf_pred, subset_new_patient, alpha=alpha
        )

        # neighbors whose conformal set excludes the chosen label
        pred_sets_names = [list(class_names[mask]) for mask in prediction_sets]
        idxs_excluding = [i for i, ps in enumerate(pred_sets_names) if label_to_exclude not in ps]

        if len(idxs_excluding) == 0:
            f.write(f"  True label: {true_label} ({categories[true_label]})\n")
            f.write(f"  Label to exclude (same group): {label_to_exclude} ({categories[label_to_exclude]})\n")
            f.write(f"  qhat = {qhat:.5f}\n")
            f.write(f"  -> No neighbor prediction sets exclude label {label_to_exclude}. Skipping instance.\n\n")
            skipped_instances += 1
            continue

        # Pick one reference neighbor to anchor from (use RNG for consistency)
        idx_example_to_anchor = int(rng.choice(idxs_excluding))
        anchor_row = subset_new_patient.iloc[idx_example_to_anchor].copy()

        # Overwrite generic symptoms with the patient's observed symptoms
        anchor_row[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
        anchor_instance = anchor_row.to_numpy()

        # Build explainer (train_data = neighborhood excluding the anchor candidate row)
        indices_train = [i for i in range(len(subset_new_patient)) if i != idx_example_to_anchor]
        explainer_orig = AnchorTabularExplainer(
            class_names=class_names,
            feature_names=subset_new_patient.columns.tolist(),
            train_data=subset_new_patient.iloc[indices_train].to_numpy(),
            discretizer=None,
            categorical_names={}
        )

        f.write(f"  True label: {true_label} ({categories[true_label]})\n")
        f.write(f"  Label to exclude (same group): {label_to_exclude} ({categories[label_to_exclude]})\n")
        f.write(f"  qhat = {qhat:.5f} | #neighbors excluding label = {len(idxs_excluding)}\n")
        f.write(f"  Anchor neighbor index in subset: {idx_example_to_anchor}\n\n")

        # Run across beam sizes
        for b in beam_sizes:
            t0 = time.perf_counter()

            exp_orig, valid_anchors_orig = explainer_orig.explain_instance(
                anchor_instance,
                xgb_cl,
                mode="conformal",
                query_label=label_to_exclude,
                qhat=qhat,
                threshold=0.95,
                delta=0.1,
                tau=0.15,
                beam_size=b,
                predicate_mode="original",
                mean_instances=None
            )

            t1 = time.perf_counter()

            cov = float(exp_orig.coverage())
            rt = float(t1 - t0)

            coverage_by_beam[b].append(cov)
            runtime_by_beam[b].append(rt)

            f.write(f"    beam={b:>2} | coverage={cov:.5f} | runtime={rt:.4f}s\n")

        f.write("\n")

    # -----------------------------
    # Aggregate statistics
    # -----------------------------
    def mean_std(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.nan, np.nan
        return float(np.mean(arr)), float(np.std(arr))

    mean_cov, std_cov, mean_rt, std_rt = [], [], [], []

    for b in beam_sizes:
        m, s = mean_std(coverage_by_beam[b])
        mean_cov.append(m)
        std_cov.append(s)

        m, s = mean_std(runtime_by_beam[b])
        mean_rt.append(m)
        std_rt.append(s)

    f.write("\n" + "=" * 100 + "\n")
    f.write("AGGREGATE STATISTICS ACROSS INSTANCES\n")
    f.write("=" * 100 + "\n\n")

    f.write(f"Used {n_instances - skipped_instances} instances; skipped {skipped_instances}.\n\n")

    for i, b in enumerate(beam_sizes):
        f.write(
            f"Beam size {b:>2}: "
            f"mean coverage = {mean_cov[i]:.5f} (std={std_cov[i]:.5f}), "
            f"mean runtime = {mean_rt[i]:.5f}s (std={std_rt[i]:.5f}s)\n"
        )

    f.write("\n")

print(f"Done. Full report saved to: {output_path}")

# -----------------------------
# Plot 1: Coverage vs beam size
# -----------------------------
plt.figure(figsize=(7, 5))

# individual traces (light)
max_traces = max(len(coverage_by_beam[b]) for b in beam_sizes)
for i in range(max_traces):
    y = []
    ok = True
    for b in beam_sizes:
        if i >= len(coverage_by_beam[b]):
            ok = False
            break
        y.append(coverage_by_beam[b][i])
    if ok:
        plt.plot(beam_sizes, y, alpha=0.2)

# mean curve
plt.plot(beam_sizes, mean_cov, marker='o', linewidth=2, label='Mean coverage')

# ---- AXIS CUSTOMIZATION ----
plt.xlim(1, 20)
plt.xticks(np.arange(1, 21, 1))                 # show 1..20 on x-axis
plt.yticks(np.arange(0, 1.01, 0.1))             # coverage ticks every 0.1
# plt.ylim(0, 1)
plt.ylim(0, 0.7)

plt.xlabel("Beam size B")
plt.ylabel("Coverage of main anchor")
plt.title("Coverage vs Beam size (One-shot mode)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("coverage_vs_beamsize_oneshot_final.pdf")
plt.show()


# -----------------------------
# Plot 2: Runtime vs Beam size (0–15 seconds)
# -----------------------------
plt.figure(figsize=(7, 5))

# individual traces (light)
max_traces_rt = max(len(runtime_by_beam[b]) for b in beam_sizes)
for i in range(max_traces_rt):
    y = []
    ok = True
    for b in beam_sizes:
        if i >= len(runtime_by_beam[b]):
            ok = False
            break
        y.append(runtime_by_beam[b][i])
    if ok:
        plt.plot(beam_sizes, y, alpha=0.2)

# mean curve
plt.plot(beam_sizes, mean_rt, marker='o', linewidth=2, label='Mean runtime')

# ---- AXIS CUSTOMIZATION ----
plt.xlim(1, 20)
plt.xticks(np.arange(1, 21, 1))                 # show 1..20 on x-axis
plt.yticks(np.arange(0, 21, 1))                 # ticks every 1 second
plt.ylim(0, 20)

plt.xlabel("Beam size B")
plt.ylabel("Runtime per explanation (seconds)")
plt.title("Runtime vs Beam size (One-shot mode)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("runtime_vs_beamsize_oneshot_final_new.pdf")
plt.show()


# -----------------------------
# Plot 3: Trade-off (Coverage vs Runtime)
# Each point is a beam size B
# -----------------------------
plt.figure(figsize=(7, 5))

mean_cov_arr = np.array(mean_cov, dtype=float)
mean_rt_arr = np.array(mean_rt, dtype=float)
beam_arr = np.array(beam_sizes, dtype=int)

# line + points (ordered by beam size already)
plt.plot(mean_rt_arr, mean_cov_arr, marker='o', linewidth=2)

for i, (b, x, y) in enumerate(zip(beam_arr, mean_rt_arr, mean_cov_arr)):
    if i < len(mean_cov_arr) - 1:
        slope = mean_cov_arr[i+1] - y
    else:
        slope = y - mean_cov_arr[i-1]

    if slope >= 0:
        offset = (6, 8)
        va = 'bottom'
    else:
        offset = (6, -10)
        va = 'top'

    plt.annotate(
        f"B={b}",
        (x, y),
        textcoords="offset points",
        xytext=offset,
        ha='left',
        va=va,
        fontsize=9
    )


# # annotate each point with its B
# for b, x, y in zip(beam_arr, mean_rt_arr, mean_cov_arr):
#     plt.annotate(f"B={b}", (x, y), textcoords="offset points", xytext=(6, 6), ha='left', fontsize=9)

# optional: make axes nicer
plt.yticks(np.arange(0, 1.01, 0.1))
# plt.ylim(0, 1)
plt.ylim(0, 0.7)
plt.xticks(np.arange(1, 12, 1)) 

plt.xlabel("Mean runtime per explanation (seconds)")
plt.ylabel("Mean coverage of main anchor")
plt.title("Coverage–runtime trade-off across beam sizes", pad=20)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("tradeoff_coverage_vs_runtime_final.pdf")
plt.show()