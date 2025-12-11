# ---------------------------------------------------------
# Run experiment on 5 test instances
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

# Compute k-nearest subsets for each test instance
all_subsets, y_subsets, similarities_list, indices_neighbors = get_knn_subsets(
    X_test, X_anchors, y_anchors, k=100
)

# --- we only want 5 test instances ---
n_instances = 5
n_instances = min(n_instances, len(X_test))   # safety
beam_size = 25

# Choose which test indices to use (here: random without replacement)
selected_test_indices = np.random.choice(len(X_test), size=n_instances, replace=False)

# To store per-(instance, label, mode) stats
results = []

output_path = "anchor_experiment_results_25_instances_all_labels.txt"
with open(output_path, "w") as f:
    f.write("EXPERIMENT ON ANCHORS (original vs mean-instances)\n")
    f.write(f"Number of test instances: {n_instances}\n")
    f.write(f"Beam size: {beam_size}\n")
    f.write("For each instance, all labels in the same disease group (excluding the true one) are used as labels-to-exclude.\n")
    f.write("=" * 100 + "\n\n")

    labels_stomach = [1, 5, 6, 7, 8]
    labels_lung = [0, 2, 3, 4, 9]

    for run_id, new_patient_idx in enumerate(selected_test_indices, 1):
        f.write(f"### INSTANCE {run_id}/{n_instances}  (test index = {new_patient_idx})\n")
        f.write("-" * 80 + "\n")

        # -----------------------------
        # 1) BASIC INSTANCE INFO
        # -----------------------------
        new_patient = X_test.iloc[new_patient_idx]
        new_patient_true_label = y_test.iloc[new_patient_idx]
        new_patient_pred = xgb_cl.predict(new_patient.values.reshape(1, -1))[0]

        f.write(f"True label: {new_patient_true_label} ({categories[new_patient_true_label]})\n")
        f.write(f"Predicted label: {new_patient_pred} ({categories[new_patient_pred]})\n\n")

        # Subset + raw subset for this instance
        subset_new_patient = all_subsets[new_patient_idx]
        y_subset_new_patient = y_subsets[new_patient_idx]

        if new_patient_idx == 10510:
            X_test_cf = X_test.copy()

            row_pos = new_patient_idx

            # Get column positions
            cough_col = X_test_cf.columns.get_loc('cough_severity')
            chest_col = X_test_cf.columns.get_loc('chest_pain_severity')

            # Assign using iloc (positional)
            X_test_cf.iloc[row_pos, cough_col] = 1
            X_test_cf.iloc[row_pos, chest_col] = 0

            # Now this will see the changed values
            new_patient_imputed = X_test_cf.iloc[row_pos]
            print("Imputed new patient:\n", new_patient_imputed[['cough_severity', 'chest_pain_severity']])

            f.write(f"Imputed new patient:\n{new_patient_imputed}\n")

            all_subsets, y_subsets, similarities_list, indices_neighbors = get_knn_subsets(X_test_cf, X_anchors, y_anchors, k=100)
            subset_new_patient = all_subsets[new_patient_idx]
            y_subset_new_patient = y_subsets[new_patient_idx]



        subset_new_patient_raw = X_anchors_orig.iloc[
            indices_neighbors[new_patient_idx]
        ].reset_index(drop=True)
        y_subset_new_patient_raw = y_anchors_orig.iloc[
            indices_neighbors[new_patient_idx]
        ].reset_index(drop=True)

        # -----------------------------
        # 2) CONFORMAL PREDICTION SETS
        # -----------------------------
        alpha = 0.01
        prediction_sets, qhat = compute_conformal_prediction_set_batch(
            xgb_cl,
            X_conf_pred,
            y_conf_pred,
            subset_new_patient,
            alpha=alpha
        )

        f.write(f"qhat: {qhat:.5f}\n")
        f.write(f"Prediction sets shape: {prediction_sets.shape}\n\n")

        # -----------------------------
        # 3) DETERMINE LABELS TO EXCLUDE FOR THIS INSTANCE
        # -----------------------------
        if new_patient_true_label in labels_stomach:
            group_labels = labels_stomach
        elif new_patient_true_label in labels_lung:
            group_labels = labels_lung
        else:
            # Fallback: all labels except the true one
            group_labels = [int(l) for l in class_names if int(l) != int(new_patient_true_label)]

        labels_to_exclude = [l for l in group_labels if l != new_patient_true_label]

        f.write(f"Labels considered for exclusion (same group): {labels_to_exclude}\n\n")

        # Precompute prediction set names once (independent of label_to_exclude)
        pred_sets_names = [list(class_names[mask]) for mask in prediction_sets]

        # -----------------------------
        # LOOP OVER ALL LABELS TO EXCLUDE
        # -----------------------------
        for label_to_exclude in labels_to_exclude:
            f.write(f"--- LABEL TO EXCLUDE: {label_to_exclude} ({categories[label_to_exclude]}) ---\n")

            # Filter prediction sets that do NOT contain this label_to_exclude
            pred_set_without_target = {}
            for i, pred_set in enumerate(pred_sets_names):
                if label_to_exclude not in pred_set:
                    pred_set_without_target[i] = pred_set

            if len(pred_set_without_target) == 0:
                f.write(f"No prediction sets exclude label {label_to_exclude}. Skipping this label.\n\n")
                continue

            f.write(f"{len(pred_set_without_target.keys())} prediction sets excluding class {label_to_exclude}:\n")
            f.write(str(pred_set_without_target) + "\n")

            # -----------------------------
            # 4) CHOOSE ANCHOR INSTANCE WITHIN SUBSET
            # -----------------------------
            idx_example_to_anchor = np.random.choice(list(pred_set_without_target.keys()))

            anchor_row = subset_new_patient.iloc[idx_example_to_anchor].copy()
            anchor_row[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
            anchor_instance = anchor_row.to_numpy()

            f.write(f"Anchor candidate index within subset: {idx_example_to_anchor}\n")
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
            # 6) BUILD EXPLAINER (once per label_to_exclude, since anchor index changes)
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
            f.write("Main anchor: %s\n" % (' AND '.join(exp_original.names())))
            f.write("Precision: %.3f\n" % exp_original.precision())
            f.write("Coverage: %.5f\n" % exp_original.coverage())
            f.write("Cumulative coverage: %.5f\n" % exp_original.cumulative_coverage())

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
                f.write(f"  {i}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}\n")

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
            f.write("Main anchor: %s\n" % (' AND '.join(exp_mean.names())))
            f.write("Precision: %.3f\n" % exp_mean.precision())
            f.write("Coverage: %.5f\n" % exp_mean.coverage())
            f.write("Cumulative coverage: %.5f\n" % exp_mean.cumulative_coverage())

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
                f.write(f"  {i}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}\n")

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
            })

            f.write("-" * 80 + "\n\n")

    # ---------------------------------------------------------
    # 9) GLOBAL STATISTICS ACROSS ALL (instance, label) PAIRS
    # ---------------------------------------------------------
    f.write("\n" + "=" * 100 + "\n")
    f.write("GLOBAL STATISTICS ACROSS ALL (INSTANCE, LABEL_TO_EXCLUDE) PAIRS\n")
    f.write("=" * 100 + "\n\n")

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

        avg_cov, std_cov = np.mean(cov_values), np.std(cov_values)
        avg_prec, std_prec = np.mean(prec_values), np.std(prec_values)
        avg_cum_cov, std_cum_cov = np.mean(cum_cov_values), np.std(cum_cov_values)
        avg_feats_valid, std_feats_valid = np.mean(feats_values), np.std(feats_values)
        avg_unique_feats_valid, std_unique_feats_valid = np.mean(unique_feats_values), np.std(unique_feats_values)
        avg_num_valid_anchors, std_num_valid_anchors = np.mean(valid_anchors_values), np.std(valid_anchors_values)

        f.write(f"MODE: {mode}\n")
        f.write(f"  Avg main precision: {avg_prec:.4f} (std={std_prec:.4f})\n")
        f.write(f"  Avg main coverage: {avg_cov:.4f} (std={std_cov:.4f})\n")
        f.write(f"  Avg cumulative coverage: {avg_cum_cov:.4f} (std={std_cum_cov:.4f})\n")
        f.write(f"  Avg #features per valid anchor: {avg_feats_valid:.4f} (std={std_feats_valid:.4f})\n")
        f.write(f"  Avg #unique features per instance (valid anchors): {avg_unique_feats_valid:.4f} (std={std_unique_feats_valid:.4f})\n")
        f.write(f"  Avg #valid anchors per instance: {avg_num_valid_anchors:.4f} (std={std_num_valid_anchors:.4f})\n")
        f.write("\n")

end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"\nDone. Full report saved to: {output_path}")
