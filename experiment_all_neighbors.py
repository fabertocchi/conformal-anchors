# ---------------------------------------------------------
# Experiment: Exclude TRUE label, using ALL neighbors
# + Check whether anchors apply to the patient instance
# ---------------------------------------------------------

from anchor.anchor.anchor_tabular import * 
from model_training import (
    X_train, y_train, X_conf_pred, y_conf_pred,
    X_anchors, y_anchors, X_test, y_test,
    X_train_orig, y_train_orig,
    X_conf_pred_orig, y_conf_pred_orig,
    X_anchors_orig, y_anchors_orig,
    X_test_orig, y_test_orig,
    categories
)
import xgboost as xgb
from subset import get_knn_subsets
from prediction_set import compute_conformal_prediction_set_batch
import matplotlib.pyplot as plt
from data.tests_v2 import TESTS
from binning import bin_dataset
import pandas as pd
import numpy as np
import time


# ---------------------------------------------------------
# Utility: check if an anchor (list of predicate names)
#          applies to a given patient instance (Series)
# ---------------------------------------------------------
def anchor_applies_to_instance(predicate_names, patient_series):
    """
    predicate_names: list of strings like
        "pulmonary_function > 54.50", "chest_xray_score ≤ 0.00", ...
    patient_series: pandas Series, features indexed by column name.

    Returns True if ALL predicates are satisfied by the patient.
    """
    for cond in predicate_names:
        cond = cond.strip()

        # Determine operator and split
        if "≤" in cond:
            feature, thresh = cond.split("≤", 1)
            op = "leq"
        elif "<=" in cond:
            feature, thresh = cond.split("<=", 1)
            op = "leq"
        elif ">" in cond:
            feature, thresh = cond.split(">", 1)
            op = "gt"
        elif "=" in cond:
            feature, thresh = cond.split("=", 1)
            op = "eq"
        else:
            # Unknown pattern -> be conservative: treat as not satisfied
            return False

        feature = feature.strip()
        thresh = thresh.strip()

        # If feature not present, treat as not satisfied
        if feature not in patient_series.index:
            return False

        x = patient_series[feature]

        # If value is NaN, we treat condition as not satisfied
        if pd.isna(x):
            return False

        # Try to parse threshold as float; if fails, fall back to string equality
        try:
            t_val = float(thresh)
            numeric = True
        except ValueError:
            numeric = False

        if op == "leq":
            if not numeric:
                return False
            if not (x <= t_val + 1e-8):
                return False
        elif op == "gt":
            if not numeric:
                return False
            if not (x > t_val - 1e-8):
                return False
        elif op == "eq":
            if numeric:
                if not (x == t_val):
                    return False
            else:
                if str(x) != thresh:
                    return False
        else:
            return False

    # All predicates satisfied
    return True


# ---------------------------------------------------------
# 0) Setup
# ---------------------------------------------------------
np.random.seed(1)

# Load the trained model
xgb_cl = xgb.XGBClassifier()
xgb_cl.load_model("xgb_model.json")
# Get class names
class_names = xgb_cl.classes_

generic_symptoms_cols = [
    'fever_severity', 'cough_severity', 'chest_pain_severity',
    'abdominal_pain_severity', 'fatigue_level', 'nausea'
]

start_time = time.time()

# Compute k-nearest subsets for each test instance (in binned space)
all_subsets, y_subsets, similarities_list, indices_neighbors = get_knn_subsets(
    X_test, X_anchors, y_anchors, k=100
)

# --- number of test instances to use ---
n_instances = 100               # <-- change if you want more
n_instances = min(n_instances, len(X_test))   # safety
beam_size = 20

# Choose which test indices to use (random without replacement)
selected_test_indices = np.random.choice(len(X_test), size=n_instances, replace=False)

# To store per-(instance, mode) stats
results = []

output_path = "anchor_experiment_true_label_exclusion_ALL_neighbors_20_with_applicability.txt"

with open(output_path, "w") as f:
    f.write("EXPERIMENT: exclude TRUE label, using ALL neighbors\n")
    f.write("Also checking if anchors apply to the patient instance.\n")
    f.write(f"Number of test instances: {n_instances}\n")
    f.write(f"Beam size: {beam_size}\n")
    f.write("=" * 100 + "\n\n")
    counter = 0   # counts instances where no prediction set excludes the true label

    for run_id, new_patient_idx in enumerate(selected_test_indices, 1):
        f.write(f"### INSTANCE {run_id}/{n_instances}  (test index = {new_patient_idx})\n")
        f.write("-" * 80 + "\n")

        # -----------------------------
        # 1) BASIC INSTANCE INFO
        # -----------------------------
        # This is the "true" patient instance we will test anchors on
        new_patient = X_test.iloc[new_patient_idx]
        new_patient_true_label = y_test.iloc[new_patient_idx]
        new_patient_pred = xgb_cl.predict(new_patient.values.reshape(1, -1))[0]

        f.write(f"True label: {new_patient_true_label} ({categories[new_patient_true_label]})\n")
        f.write(f"Predicted label: {new_patient_pred} ({categories[new_patient_pred]})\n\n")

        # KNN subset in binned space + raw space for this instance (ALL neighbors)
        subset_new_patient = all_subsets[new_patient_idx]     # binned features
        y_subset_new_patient = y_subsets[new_patient_idx]

        subset_new_patient_raw = X_anchors_orig.iloc[
            indices_neighbors[new_patient_idx]
        ].reset_index(drop=True)
        y_subset_new_patient_raw = y_anchors_orig.iloc[
            indices_neighbors[new_patient_idx]
        ].reset_index(drop=True)

        f.write(f"Total neighbors (all labels): {len(y_subset_new_patient)}\n")

        # -----------------------------
        # 2) CONFORMAL PREDICTION SETS ON ALL NEIGHBORS
        # -----------------------------
        alpha = 0.01
        prediction_sets, qhat = compute_conformal_prediction_set_batch(
            xgb_cl,
            X_conf_pred,
            y_conf_pred,
            subset_new_patient,   # <-- ALL neighbors, not filtered
            alpha=alpha
        )

        f.write(f"qhat: {qhat:.5f}\n")
        f.write(f"Prediction sets shape (all neighbors): {prediction_sets.shape}\n\n")

        # -----------------------------
        # 3) LABEL TO EXCLUDE = TRUE LABEL
        # -----------------------------
        label_to_exclude = int(new_patient_true_label)
        f.write(f"Label to exclude (TRUE label): {label_to_exclude} ({categories[label_to_exclude]})\n\n")

        # Prediction set names
        pred_sets_names = [list(class_names[mask]) for mask in prediction_sets]

        pred_set_without_target = {}
        for i, pred_set in enumerate(pred_sets_names):
            if label_to_exclude not in pred_set:
                pred_set_without_target[i] = pred_set

        if len(pred_set_without_target) == 0:
            f.write("No prediction sets exclude the TRUE label\n")
            f.write("-" * 80 + "\n\n")
            counter += 1
            continue

        f.write(f"{len(pred_set_without_target.keys())} prediction sets excluding TRUE label {label_to_exclude}:\n")
        f.write(str(pred_set_without_target) + "\n\n")

        # -----------------------------
        # 4) CHOOSE ANCHOR INSTANCE WITHIN FULL SUBSET
        # -----------------------------
        idx_example_to_anchor = np.random.choice(list(pred_set_without_target.keys()))

        # Start from this neighbor row in binned space
        anchor_row = subset_new_patient.iloc[idx_example_to_anchor].copy()
        # overwrite generic symptoms with new patient's symptoms
        anchor_row[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
        anchor_instance = anchor_row.to_numpy()

        f.write(f"Anchor candidate index within full subset: {idx_example_to_anchor}\n")
        f.write(f"True label of (original) anchor neighbor: {y_subset_new_patient.iloc[idx_example_to_anchor]}\n")
        f.write(f"Predicted label of modified anchor instance: {xgb_cl.predict(anchor_instance.reshape(1, -1))[0]}\n\n")

        idxs_neighbors_excluding_target = list(pred_set_without_target.keys())
        neighbors_excluding_target = subset_new_patient.iloc[idxs_neighbors_excluding_target]
        neighbors_excluding_target_raw = subset_new_patient_raw.iloc[idxs_neighbors_excluding_target]
        neighbors_labels = y_subset_new_patient.iloc[idxs_neighbors_excluding_target]

        # -----------------------------
        # 5) MEAN INSTANCES (for mean-instances mode)
        # -----------------------------
        unique_labels = np.unique(neighbors_labels)
        mean_instances_raw = []
        for label in unique_labels:
            mask = (neighbors_labels == label)
            cluster = neighbors_excluding_target_raw[mask]
            mean_instances_raw.append(cluster.mean(axis=0))
        mean_instances_raw = np.vstack(mean_instances_raw)

        mean_instances_df = pd.DataFrame(
            mean_instances_raw,
            columns=X_train_orig.columns
        )
        mean_instances_binned_df = bin_dataset(
            mean_instances_df,
            TESTS=TESTS,
            generic_symptoms_cols=generic_symptoms_cols,
            verbose=False
        )
        # overwrite generic symptoms with new patient's symptoms
        mean_instances_binned_df[generic_symptoms_cols] = (
            new_patient[generic_symptoms_cols].values
        )

        # -----------------------------
        # 6) BUILD EXPLAINER (train on ALL neighbors)
        # -----------------------------
        feature_cols = X_train.columns.tolist()
        categorical_names = {}
        indices = [i for i in range(len(subset_new_patient)) if i != idx_example_to_anchor]

        explainer = AnchorTabularExplainer(
            class_names=class_names,
            feature_names=feature_cols,
            train_data=subset_new_patient.iloc[indices].to_numpy(),
            discretizer=None,
            categorical_names=categorical_names,
        )

        # ----------------------------------------------------
        # 7) ORIGINAL MODE
        # ----------------------------------------------------
        exp_original, valid_anchors_original = explainer.explain_instance(
            anchor_instance, xgb_cl, mode="conformal",
            query_label=label_to_exclude, qhat=qhat,
            threshold=0.95, delta=0.1, tau=0.15, beam_size=beam_size,
            predicate_mode="original"
        )

        f.write("ORIGINAL MODE\n")
        main_names_orig = exp_original.names()
        f.write("Main anchor: %s\n" % (' AND '.join(main_names_orig)))
        f.write("Precision: %.3f\n" % exp_original.precision())
        f.write("Coverage: %.5f\n" % exp_original.coverage())
        f.write("Cumulative coverage: %.5f\n" % exp_original.cumulative_coverage())
        f.write(f"---> valid anchors found: {len(valid_anchors_original)} {[d['feature'] for d in valid_anchors_original]}\n")

        # Check applicability of main and valid anchors to the patient instance
        main_applies_orig = anchor_applies_to_instance(main_names_orig, new_patient)
        f.write(f"Does MAIN anchor (original) apply to patient? {main_applies_orig}\n")

        num_valid_apply_orig = 0
        for va in valid_anchors_original:
            if anchor_applies_to_instance(va['names'], new_patient):
                num_valid_apply_orig += 1
        f.write(f"#valid anchors (original) applying to patient: {num_valid_apply_orig} / {len(valid_anchors_original)}\n")

        if len(valid_anchors_original) > 0:
            num_feats_each = [len(va['feature']) for va in valid_anchors_original]
            avg_feats = float(np.mean(num_feats_each))
            unique_feats = set()
            for va in valid_anchors_original:
                unique_feats |= set(va['feature'])
            num_unique_feats = len(unique_feats)
        else:
            avg_feats = 0.0
            num_unique_feats = 0

        f.write(f"Valid anchors (original): {len(valid_anchors_original)}\n")
        f.write(f"Avg #features per valid anchor (original): {avg_feats:.3f}\n")
        f.write(f"#unique features across valid anchors (original): {num_unique_feats}\n")

        sorted_valid_anchors_original = sorted(
            valid_anchors_original,
            key=lambda va: va['coverage'][-1],
            reverse=True
        )
        for i, va in enumerate(sorted_valid_anchors_original, 1):
            names = " AND ".join(va['names'])
            prec = va['precision'][-1]
            cov = va['coverage'][-1]
            applies = anchor_applies_to_instance(va['names'], new_patient)
            f.write(f"  {i}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}, applies_to_patient={applies}\n")

        f.write("\n")

        results.append({
            'instance_idx': new_patient_idx,
            'label_to_exclude': label_to_exclude,
            'mode': 'original',
            'main_precision': exp_original.precision(),
            'main_coverage': exp_original.coverage(),
            'cumulative_coverage': exp_original.cumulative_coverage(),
            'avg_feats_valid': avg_feats,
            'num_unique_feats_valid': num_unique_feats,
            'num_valid_anchors': len(valid_anchors_original),
            'main_applies': int(main_applies_orig),
            'num_valid_apply': num_valid_apply_orig,
            'no_anchor_applies': int((not main_applies_orig) and (num_valid_apply_orig == 0)),
        })

        # ----------------------------------------------------
        # 8) MEAN-INSTANCES MODE
        # ----------------------------------------------------
        exp_mean, valid_anchors_mean = explainer.explain_instance(
            anchor_instance,
            xgb_cl,
            mode="conformal",
            query_label=label_to_exclude,
            qhat=qhat,
            threshold=0.95,
            delta=0.1,
            tau=0.15,
            beam_size=beam_size,
            predicate_mode="mean_instances",
            mean_instances=mean_instances_binned_df.to_numpy()
        )

        f.write("MEAN-INSTANCES MODE\n")
        main_names_mean = exp_mean.names()
        f.write("Main anchor: %s\n" % (' AND '.join(main_names_mean)))
        f.write("Precision: %.3f\n" % exp_mean.precision())
        f.write("Coverage: %.5f\n" % exp_mean.coverage())
        f.write("Cumulative coverage: %.5f\n" % exp_mean.cumulative_coverage())
        f.write(f"---> valid anchors found: {len(valid_anchors_mean)} {[d['feature'] for d in valid_anchors_mean]}\n")

        # Check applicability for mean-instances mode
        main_applies_mean = anchor_applies_to_instance(main_names_mean, new_patient)
        f.write(f"Does MAIN anchor (mean-instances) apply to patient? {main_applies_mean}\n")

        num_valid_apply_mean = 0
        for va in valid_anchors_mean:
            if anchor_applies_to_instance(va['names'], new_patient):
                num_valid_apply_mean += 1
        f.write(f"#valid anchors (mean) applying to patient: {num_valid_apply_mean} / {len(valid_anchors_mean)}\n")

        if len(valid_anchors_mean) > 0:
            num_feats_each_m = [len(va['feature']) for va in valid_anchors_mean]
            avg_feats_m = float(np.mean(num_feats_each_m))
            unique_feats_m = set()
            for va in valid_anchors_mean:
                unique_feats_m |= set(va['feature'])
            num_unique_feats_m = len(unique_feats_m)
        else:
            avg_feats_m = 0.0
            num_unique_feats_m = 0

        f.write(f"Valid anchors (mean-instances): {len(valid_anchors_mean)}\n")
        f.write(f"Avg #features per valid anchor (mean): {avg_feats_m:.3f}\n")
        f.write(f"#unique features across valid anchors (mean): {num_unique_feats_m}\n")

        sorted_valid_anchors_mean = sorted(
            valid_anchors_mean,
            key=lambda va: va['coverage'][-1],
            reverse=True
        )
        for i, va in enumerate(sorted_valid_anchors_mean, 1):
            names = " AND ".join(va['names'])
            prec = va['precision'][-1]
            cov = va['coverage'][-1]
            applies = anchor_applies_to_instance(va['names'], new_patient)
            f.write(f"  {i}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}, applies_to_patient={applies}\n")

        f.write("\n")

        results.append({
            'instance_idx': new_patient_idx,
            'label_to_exclude': label_to_exclude,
            'mode': 'mean',
            'main_precision': exp_mean.precision(),
            'main_coverage': exp_mean.coverage(),
            'cumulative_coverage': exp_mean.cumulative_coverage(),
            'avg_feats_valid': avg_feats_m,
            'num_unique_feats_valid': num_unique_feats_m,
            'num_valid_anchors': len(valid_anchors_mean),
            'main_applies': int(main_applies_mean),
            'num_valid_apply': num_valid_apply_mean,
            'no_anchor_applies': int((not main_applies_mean) and (num_valid_apply_mean == 0)),
        })

        f.write("-" * 80 + "\n\n")

    # ---------------------------------------------------------
    # 9) GLOBAL STATISTICS
    # ---------------------------------------------------------
    f.write("\n" + "=" * 100 + "\n")
    f.write("GLOBAL STATISTICS ACROSS ALL (INSTANCE, TRUE_LABEL_EXCLUSION) PAIRS\n")
    f.write("=" * 100 + "\n\n")

    f.write(f"Number of instances with no neighbors' prediction sets excluding true label: {counter}\n\n")

    for mode in ['original', 'mean']:
        mode_results = [r for r in results if r['mode'] == mode]
        if len(mode_results) == 0:
            continue

        cov_values = [r['main_coverage'] for r in mode_results]
        prec_values = [r['main_precision'] for r in mode_results]
        cum_cov_values = [r['cumulative_coverage'] for r in mode_results]
        feats_values = [r['avg_feats_valid'] for r in mode_results]
        unique_feats_values = [r['num_unique_feats_valid'] for r in mode_results]
        valid_anchors_values = [r['num_valid_anchors'] for r in mode_results]
        main_applies_values = [r['main_applies'] for r in mode_results]
        num_valid_apply_values = [r['num_valid_apply'] for r in mode_results]
        no_anchor_applies_values = [r['no_anchor_applies'] for r in mode_results]
        count_no_anchor_applies = int(np.sum(no_anchor_applies_values))
        frac_no_anchor_applies = count_no_anchor_applies / len(mode_results)

        avg_cov, std_cov = np.mean(cov_values), np.std(cov_values)
        avg_prec, std_prec = np.mean(prec_values), np.std(prec_values)
        avg_cum_cov, std_cum_cov = np.mean(cum_cov_values), np.std(cum_cov_values)
        avg_feats_valid, std_feats_valid = np.mean(feats_values), np.std(feats_values)
        avg_unique_feats_valid, std_unique_feats_valid = np.mean(unique_feats_values), np.std(unique_feats_values)
        avg_num_valid_anchors, std_num_valid_anchors = np.mean(valid_anchors_values), np.std(valid_anchors_values)
        frac_main_applying = np.mean(main_applies_values)
        avg_num_valid_apply = np.mean(num_valid_apply_values)

        f.write(f"MODE: {mode}\n")
        f.write(f"  Avg main precision: {avg_prec:.4f} (std={std_prec:.4f})\n")
        f.write(f"  Avg main coverage: {avg_cov:.4f} (std={std_cov:.4f})\n")
        f.write(f"  Avg cumulative coverage: {avg_cum_cov:.4f} (std={std_cum_cov:.4f})\n")
        f.write(f"  Avg #features per valid anchor: {avg_feats_valid:.4f} (std={std_feats_valid:.4f})\n")
        f.write(f"  Avg #unique features per instance (valid anchors): {avg_unique_feats_valid:.4f} (std={std_unique_feats_valid:.4f})\n")
        f.write(f"  Avg #valid anchors per instance: {avg_num_valid_anchors:.4f} (std={std_num_valid_anchors:.4f})\n")
        f.write(f"  Fraction of MAIN anchors applying to patient: {frac_main_applying:.4f}\n")
        f.write(f"  Avg #valid anchors applying to patient: {avg_num_valid_apply:.4f}\n")
        f.write(
        f"  #instances with NO anchors (main nor any valid) applying to patient: "
        f"{count_no_anchor_applies} / {len(mode_results)}\n"
        )
        f.write(
            f"  Fraction of instances with NO anchors applying: "
            f"{frac_no_anchor_applies:.4f}\n"
        )
        f.write("\n")

end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"\nDone. Full report saved to: {output_path}")
