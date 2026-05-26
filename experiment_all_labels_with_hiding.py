# ---------------------------------------------------------
# K-MEDOIDS + MEAN-INSTANCES EXPERIMENT ON 20 TEST INSTANCES
# Original vs Mean-Instances vs Medoid (+ Union Modes 1 & 2)
# With Missing Generic Symptoms (Randomly Hidden 1–3)
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
from subset import get_knn_subsets, gower_similarity_mixed_raw
from prediction_set import compute_conformal_prediction_set_batch
from data.tests_v2 import TESTS
from binning import bin_dataset
from dataset_basic_setting import TEST_BOUNDS
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

np.random.seed(1)

# ---------------------------------------------------------
# Helper: check if an anchor applies to a given instance
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

        if feature not in patient_series.index:
            return False

        x = patient_series[feature]

        # Missing value => condition not satisfied
        if pd.isna(x):
            return False

        try:
            t_val = float(thresh)
            numeric = True
        except ValueError:
            numeric = False

        if op == "leq":
            if not numeric or not (x <= t_val):
                return False
        elif op == "gt":
            if not numeric or not (x > t_val):
                return False
        elif op == "eq":
            if numeric:
                if not (abs(x - t_val) < 1e-20):
                    return False
            else:
                if str(x) != thresh:
                    return False
        else:
            return False

    return True


# ---------------------------------------------------------
# Gower distance helpers (RAW)
# ---------------------------------------------------------
def gower_distance_to_all_raw(X, x, feature_names, symptom_cols, TEST_BOUNDS, symptom_range=(0, 10)):
    # distance = 1 - similarity
    return 1.0 - gower_similarity_mixed_raw(
        anchor_vectors=X,
        test_vector=x,
        feature_names=feature_names,
        symptom_cols=symptom_cols,
        TEST_BOUNDS=TEST_BOUNDS,
        feature_range=symptom_range
    )


def compute_gower_distance_matrix_raw(X, feature_names, symptom_cols, TEST_BOUNDS, symptom_range=(0, 10)):
    """
    Precompute full (n x n) distance matrix using RAW Gower.
    Complexity O(n^2 d).
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        D[i, :] = gower_distance_to_all_raw(
            X, X[i],
            feature_names=feature_names,
            symptom_cols=symptom_cols,
            TEST_BOUNDS=TEST_BOUNDS,
            symptom_range=symptom_range
        )
    return D


# ---------------------------------------------------------
# UNION PRUNING – MODE 1 (max marginal gain each step)
# ---------------------------------------------------------
def union_prune_anchors_mode1(anchors, coverage_df, n_samples=10000, flatten_tol=1e-6):
    if not anchors:
        return [], 0.0, [], []

    n_universe = coverage_df.shape[0]
    if n_universe == 0:
        return [], 0.0, [], []

    sampled_idx = np.random.choice(range(n_universe), size=n_samples, replace=True)
    coverage_data = coverage_df.iloc[sampled_idx].reset_index(drop=True)
    n_cov = coverage_data.shape[0]

    state = {'t_coverage_idx': {}}
    for t, a in enumerate(anchors):
        mask = coverage_data.apply(
            lambda row: anchor_applies_to_instance(a['names'], row),
            axis=1
        ).to_numpy(dtype=bool)

        covered_idx = set(np.where(mask)[0])
        state['t_coverage_idx'][t] = covered_idx
        a['coverage_idx'] = covered_idx
        a['cov_train'] = float(len(covered_idx)) / n_cov

    final_anchors = []
    covered_union = set()
    last_cumulative_coverage = 0.0

    remaining = set(range(len(anchors)))
    coverage_trajectory = []
    gain_trajectory = []

    while remaining:
        best_t = None
        best_gain = 0.0
        best_candidate_union = None

        for t in remaining:
            candidate_union = covered_union | state['t_coverage_idx'][t]
            cumulative_coverage = float(len(candidate_union)) / n_cov
            gain = cumulative_coverage - last_cumulative_coverage

            if gain > best_gain:
                best_gain = gain
                best_t = t
                best_candidate_union = candidate_union

        if best_t is None or best_gain <= flatten_tol:
            break

        final_anchors.append(anchors[best_t])
        covered_union = best_candidate_union
        last_cumulative_coverage = float(len(covered_union)) / n_cov
        remaining.remove(best_t)

        coverage_trajectory.append(last_cumulative_coverage)
        gain_trajectory.append(best_gain)

    return final_anchors, last_cumulative_coverage, coverage_trajectory, gain_trajectory


# ---------------------------------------------------------
# UNION PRUNING – MODE 2 (sorted by individual coverage)
# ---------------------------------------------------------
def union_prune_anchors_mode2(anchors, coverage_df, n_samples=10000, flatten_tol=1e-6):
    if not anchors:
        return [], 0.0, [], []

    n_universe = coverage_df.shape[0]
    if n_universe == 0:
        return [], 0.0, [], []

    sampled_idx = np.random.choice(range(n_universe), size=n_samples, replace=True)
    coverage_data = coverage_df.iloc[sampled_idx].reset_index(drop=True)
    n_cov = coverage_data.shape[0]

    state = {'t_coverage_idx': {}}
    for t, a in enumerate(anchors):
        mask = coverage_data.apply(
            lambda row: anchor_applies_to_instance(a['names'], row),
            axis=1
        ).to_numpy(dtype=bool)

        covered_idx = set(np.where(mask)[0])
        state['t_coverage_idx'][t] = covered_idx
        a['coverage_idx'] = covered_idx
        a['cov_train'] = float(len(covered_idx)) / n_cov

    sorted_t = sorted(range(len(anchors)), key=lambda t: anchors[t]['cov_train'], reverse=True)

    final_anchors = []
    covered_union = set()
    last_cumulative_coverage = 0.0

    coverage_trajectory = []
    gain_trajectory = []

    for rank, t in enumerate(sorted_t, start=1):
        candidate_union = covered_union | state['t_coverage_idx'][t]
        cumulative_coverage = float(len(candidate_union)) / n_cov

        if rank == 1:
            final_anchors.append(anchors[t])
            covered_union = candidate_union
            last_cumulative_coverage = cumulative_coverage

            coverage_trajectory.append(cumulative_coverage)
            gain_trajectory.append(cumulative_coverage)
        else:
            gain = cumulative_coverage - last_cumulative_coverage
            if gain <= flatten_tol:
                break

            final_anchors.append(anchors[t])
            covered_union = candidate_union
            last_cumulative_coverage = cumulative_coverage

            coverage_trajectory.append(cumulative_coverage)
            gain_trajectory.append(gain)

    return final_anchors, last_cumulative_coverage, coverage_trajectory, gain_trajectory


generic_symptoms_cols = [
    'fever_severity', 'cough_severity', 'chest_pain_severity',
    'abdominal_pain_severity', 'fatigue_level', 'nausea'
]

LUNG_TESTS = ['pulmonary_function', 'chest_xray_score', 'sputum_neutrophil_percent', 'wbc_count']
STOMACH_TESTS =['endoscopy_score', 'gastric_ph', 'h_pylori_level', 'hemoglobin']
DIAGNOSTIC_TEST_COLS = LUNG_TESTS + STOMACH_TESTS


# ---------------------------------------------------------
# Helper: count predicate types in an anchor
# ---------------------------------------------------------
def count_predicate_types(predicate_names):
    """
    Counts how many predicates in an anchor involve:
    - generic symptoms
    - lung-related tests
    - stomach-related tests
    """
    counts = {
        'generic': 0,
        'lung': 0,
        'stomach': 0,
        'other': 0
    }

    for cond in predicate_names:
        cond = cond.strip()

        parts = cond.split(" ")

        if len(parts) == 3:
            feature = parts[0].strip()
        elif len(parts) == 5:
            feature = parts[2].strip()
        else:
            raise ValueError(f"Unexpected predicate format: '{cond}'")

        # # Extract feature name
        # feature = None
        # for op in ["≤", "<=", ">", "="]:
        #     if op in cond:
        #         feature = cond.split(op, 1)[0].strip()
        #         break

        # if feature is None:
        #     counts['other'] += 1
        #     continue

        if feature in generic_symptoms_cols:
            counts['generic'] += 1
        elif feature in LUNG_TESTS:
            counts['lung'] += 1
        elif feature in STOMACH_TESTS:
            counts['stomach'] += 1
        else:
            counts['other'] += 1

    return counts


def classify_anchor_type(predicate_names):
    """
    Classifies an anchor as:
    - 'generic': all predicates use generic symptom features
    - 'lung': at least one lung-related test, no stomach tests
    - 'stomach': at least one stomach-related test, no lung tests
    - 'mixed': mixture of lung and stomach, or involving other features
    """
    counts = count_predicate_types(predicate_names)
    total = counts['generic'] + counts['lung'] + counts['stomach'] + counts['other']

    if total == 0:
        return 'mixed'  # nothing to classify

    # Prioritize organ-specific tests over generic symptoms:
    # if there is any lung condition and no stomach → lung
    if counts['lung'] > 0 and counts['stomach'] == 0:
        return 'lung'

    # if there is any stomach condition and no lung → stomach
    if counts['stomach'] > 0 and counts['lung'] == 0:
        return 'stomach'

    # only generic symptoms (no lung / stomach / other)
    if counts['generic'] == total:
        return 'generic'

    # everything else: mixed (lung+stomach, or 'other' present)
    return 'mixed'



# ---------------------------------------------------------
# MODEL & DATA
# ---------------------------------------------------------
xgb_cl = xgb.XGBClassifier()
xgb_cl.load_model("xgb_model.json")
class_names = xgb_cl.classes_

start_time = time.time()

# # k-NN subsets for each test instance (on full X_test, before hiding)
# all_subsets, y_subsets, similarities_list, indices_neighbors = get_knn_subsets(
#     X_test, X_anchors, y_anchors, k=100
# )

# ---------------------------------------------------------
# EXPERIMENT SETTINGS
# ---------------------------------------------------------
n_instances = 5
n_instances = min(n_instances, len(X_test))
beam_size = 10

selected_test_indices = np.random.choice(len(X_test), size=n_instances, replace=False)

results = []

output_path = "anchor_kmedoids_mean_missing_symptoms_all_labels_NAN_5instances_10_labels.txt"
with open(output_path, "w") as f:
    f.write("K-MEDOIDS + MEAN-INSTANCES EXPERIMENT ON ANCHORS\n")
    f.write("Modes: original, mean-instances, medoid, union_mode1, union_mode2\n")
    f.write(f"Number of test instances: {n_instances}\n")
    f.write(f"Beam size: {beam_size}\n")
    f.write("For each instance, all labels in the same disease group (excluding the true one) "
            "are used as labels-to-exclude.\n")
    f.write("For each instance, between 1 and 3 generic symptoms are randomly hidden (set to NaN).\n")
    f.write("=" * 100 + "\n\n")

    labels_stomach = [1, 5, 6, 7, 8]
    labels_lung = [0, 2, 3, 4, 9]

    for run_id, new_patient_idx in enumerate(selected_test_indices, 1):
        f.write(f"### INSTANCE {run_id}/{n_instances}  (test index = {new_patient_idx})\n")
        f.write("-" * 80 + "\n")

        # -----------------------------
        # 1) BASIC INSTANCE INFO + HIDE SYMPTOMS
        # -----------------------------
        # Start from full test instance
        new_patient = X_test.iloc[new_patient_idx].copy()
        new_patient_true_label = y_test.iloc[new_patient_idx]

        # Randomly choose how many and which symptoms to hide
        n_hide = np.random.randint(1, 4)  # 1, 2, or 3
        symptoms_to_hide = list(
            np.random.choice(generic_symptoms_cols, size=n_hide, replace=False) # uniform distribution over symptoms to hide
        )

        # Hide them (set to NaN)
        for col in symptoms_to_hide:
            new_patient[col] = np.nan

        new_patient_df = new_patient.to_frame().T           # shape (1, d)
        # (optional but nice) keep the original index as the test index
        new_patient_df.index = [new_patient_idx]

        new_patient_for_pred = new_patient.copy()
        new_patient_for_pred[DIAGNOSTIC_TEST_COLS] = np.nan

        new_patient_pred_nan = xgb_cl.predict(
            new_patient_for_pred.values.reshape(1, -1)
        )[0]

        probs_nan = xgb_cl.predict_proba(
            new_patient_for_pred.values.reshape(1, -1)
        )[0]

        # Model prediction on the *masked* instance (XGBoost can handle NaNs)
        new_patient_pred = xgb_cl.predict(new_patient.values.reshape(1, -1))[0]

        probs = xgb_cl.predict_proba(new_patient.values.reshape(1, -1))[0]

        f.write(f"New patient: {new_patient}\n")
        f.write(f"True label: {new_patient_true_label} ({categories[new_patient_true_label]})\n")
        f.write(f"Predicted label (with hidden symptoms): {new_patient_pred} "
                f"({categories[new_patient_pred]})\n")
        f.write(f"Predicted label (with tests set to NaN): {new_patient_pred_nan} "
                f"({categories[new_patient_pred_nan]})\n")
        # f.write(f"Number of hidden symptoms: {len(symptoms_to_hide)}\n")
        # f.write(f"Hidden generic symptoms for this patient: {symptoms_to_hide}\n\n")
        # f.write("Model class probabilities for masked test instance:\n")
        for cls_idx, p in enumerate(probs):
            f.write(
                f"  P(y={cls_idx} | x) = {p:.5f} "
                f"({categories[int(cls_idx)]})\n"
            )

        for cls_idx, p in enumerate(probs_nan):
            f.write(
                f"  P(y={cls_idx} | x) = {p:.5f} "
                f"({categories[int(cls_idx)]})\n"
            )

        # Sum probabilities over lung-related vs stomach-related diseases
        prob_lung = float(sum(probs[l] for l in labels_lung))
        prob_stomach = float(sum(probs[l] for l in labels_stomach))

        # Sum probabilities over lung-related vs stomach-related diseases
        prob_lung_nan = float(sum(probs_nan[l] for l in labels_lung))
        prob_stomach_nan = float(sum(probs_nan[l] for l in labels_stomach))

        f.write(f"Sum of probabilities over LUNG diseases: {prob_lung:.5f}\n")
        f.write(f"Sum of probabilities over STOMACH diseases: {prob_stomach:.5f}\n\n")

        f.write(f"Sum of probabilities over LUNG diseases NAN: {prob_lung_nan:.5f}\n")
        f.write(f"Sum of probabilities over STOMACH diseases NAN: {prob_stomach_nan:.5f}\n\n")

        # # Subset (binned) + raw subset for this instance (neighbors fixed from full X_test)
        # subset_new_patient = all_subsets[new_patient_idx]
        # y_subset_new_patient = y_subsets[new_patient_idx]

        # subset_new_patient_raw = X_anchors_orig.iloc[
        #     indices_neighbors[new_patient_idx]
        # ].reset_index(drop=True)
        # y_subset_new_patient_raw = y_anchors_orig.iloc[
        #     indices_neighbors[new_patient_idx]
        # ].reset_index(drop=True)


        subsets_list, y_subsets_list, similarities_list, indices_neighbors_list = get_knn_subsets(new_patient_df, X_anchors, y_anchors, k=100)

        subset_new_patient = subsets_list[0]             # DataFrame of neighbors (binned)
        y_subset_new_patient = y_subsets_list[0]         # Series of neighbor labels
        similarity_list = similarities_list[0]           # similarities for this patient
        indices_neighbors = indices_neighbors_list[0]    # indices in X_anchors_orig / y_anchors_orig

        subset_new_patient_raw = X_anchors_orig.iloc[indices_neighbors].reset_index(drop=True)
        y_subset_new_patient_raw = y_anchors_orig.iloc[indices_neighbors].reset_index(drop=True)

        # -----------------------------
        # 1b) NEIGHBOR LABEL COMPOSITION (LUNG vs STOMACH)
        # -----------------------------
        n_neighbors = len(y_subset_new_patient)
        n_lung_neighbors = int(np.isin(y_subset_new_patient, labels_lung).sum())
        n_stomach_neighbors = int(np.isin(y_subset_new_patient, labels_stomach).sum())

        f.write("Neighbor label composition (k-NN subset):\n")
        f.write(f"  Total neighbors: {n_neighbors}\n")
        f.write(f"  Lung-disease neighbors:    {n_lung_neighbors}\n")
        f.write(f"  Stomach-disease neighbors: {n_stomach_neighbors}\n")
        f.write("\n")

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
        # 3) LABELS TO EXCLUDE (ALL IN SAME GROUP EXCEPT TRUE)
        # -----------------------------
        # if new_patient_true_label in labels_stomach:
        #     group_labels = labels_stomach
        # elif new_patient_true_label in labels_lung:
        #     group_labels = labels_lung
        # else:
        #     group_labels = [int(l) for l in class_names if int(l) != int(new_patient_true_label)]

        # labels_to_exclude = [l for l in group_labels if l != new_patient_true_label]
        labels_to_exclude = [l for l in class_names]
        f.write(f"Labels considered for exclusion (same group, all except true): "
                f"{labels_to_exclude}\n\n")

        pred_sets_names = [list(class_names[mask]) for mask in prediction_sets]

        # -----------------------------
        # LOOP OVER ALL LABELS TO EXCLUDE
        # -----------------------------
        for label_to_exclude in labels_to_exclude:
            f.write(f"--- LABEL TO EXCLUDE: {label_to_exclude} "
                    f"({categories[label_to_exclude]}) ---\n")

            # 3a) prediction sets that exclude the target label
            pred_set_without_target = {}
            for i, pred_set in enumerate(pred_sets_names):
                if label_to_exclude not in pred_set:
                    pred_set_without_target[i] = pred_set

            if len(pred_set_without_target) == 0:
                f.write(f"No prediction sets exclude label {label_to_exclude}. "
                        "Skipping this label.\n\n")
                continue

            f.write(f"{len(pred_set_without_target.keys())} prediction sets excluding "
                    f"class {label_to_exclude}:\n")
            f.write(str(pred_set_without_target) + "\n\n")

            # Neighbors excluding target
            idxs_neighbors_excluding_target = list(pred_set_without_target.keys())
            neighbors_excluding_target = subset_new_patient.iloc[idxs_neighbors_excluding_target]
            neighbors_excluding_target_raw = subset_new_patient_raw.iloc[idxs_neighbors_excluding_target]
            neighbors_labels = y_subset_new_patient.iloc[idxs_neighbors_excluding_target]
            unique_labels = np.unique(neighbors_labels)
            f.write("  Unique neighbor labels (indices): "
                    f"{unique_labels.tolist()}\n")
            f.write("  Unique neighbor labels (diseases): "
                    f"{', '.join(f'{int(l)} ({categories[int(l)]})' for l in unique_labels)}\n\n")

            # Label composition among neighbors whose prediction sets exclude this label
            n_neighbors_excl = len(neighbors_labels)
            n_lung_excl = int(np.isin(neighbors_labels, labels_lung).sum())
            n_stomach_excl = int(np.isin(neighbors_labels, labels_stomach).sum())


            f.write("Neighbor label composition (prediction sets excluding this label):\n")
            f.write(f"  Total neighbors: {n_neighbors_excl}\n")
            f.write(f"  Lung-disease neighbors:    {n_lung_excl}\n")
            f.write(f"  Stomach-disease neighbors: {n_stomach_excl}\n\n")

            # -----------------------------
            # 4) ORIGINAL MODE PREPARATION (ANCHOR INSTANCE)
            # -----------------------------
            idx_example_to_anchor = np.random.choice(idxs_neighbors_excluding_target)
            anchor_row = neighbors_excluding_target.loc[idx_example_to_anchor].copy()

            # Impute (possibly missing) generic symptoms from the masked patient
            anchor_row[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
            anchor_instance = anchor_row.to_numpy()

            f.write(f"Anchor candidate index within subset: {idx_example_to_anchor}\n")
            f.write(f"True label of anchor neighbor: "
                    f"{y_subset_new_patient.iloc[idx_example_to_anchor]}\n")
            f.write(f"Predicted label of anchor instance (with hidden symptoms): "
                    f"{xgb_cl.predict(anchor_instance.reshape(1, -1))[0]}\n\n")
            
            # Probabilities for the anchor instance
            anchor_probs = xgb_cl.predict_proba(anchor_instance.reshape(1, -1))[0]
            f.write("Model class probabilities for anchor instance:\n")
            for cls_idx, p in enumerate(anchor_probs):
                f.write(
                    f"  P(y={cls_idx} | anchor) = {p:.5f} "
                    f"({categories[int(cls_idx)]})\n"
                )

            # -----------------------------
            # 5) MEAN INSTANCES (for mean-instances mode)
            # -----------------------------
            
            mean_instances_raw = []
            cluster_list = []
            cluster_indices_list = []
            for lab in unique_labels:
                mask_lab = (neighbors_labels == lab)
                cluster = neighbors_excluding_target_raw.loc[mask_lab]
                cluster_indices = np.where(mask_lab.values)[0]
                cluster_list.append(cluster)
                cluster_indices_list.append(cluster_indices)
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
            # Overwrite generic symptoms in mean instances with masked patient's symptoms
            mean_instances_binned_df[generic_symptoms_cols] = (
                new_patient[generic_symptoms_cols].values
            )

            # -----------------------------
            # 6) K-MEDOIDS PREPARATION (RAW GOWER)
            # -----------------------------
            X_cluster_raw_df = neighbors_excluding_target_raw.reset_index(drop=True)
            X_cluster_raw = X_cluster_raw_df.to_numpy(dtype=float)
            feature_names_raw = X_cluster_raw_df.columns.tolist()

            D = compute_gower_distance_matrix_raw(
                X_cluster_raw,
                feature_names=feature_names_raw,
                symptom_cols=generic_symptoms_cols,
                TEST_BOUNDS=TEST_BOUNDS,
                symptom_range=(0, 10)
            )

            # Sanity checks for clusters (optional)
            for i, lab in enumerate(unique_labels):
                assert (neighbors_labels.loc[cluster_list[i].index] == lab).all()
            for i in range(len(unique_labels)):
                assert np.allclose(
                    cluster_list[i][feature_names_raw].to_numpy(dtype=float),
                    X_cluster_raw[cluster_indices_list[i]], equal_nan=True
                )

            # Snap each mean to nearest actual point -> init medoids indices
            init_medoids = []
            for i, m in enumerate(mean_instances_raw):
                X_cluster_label = X_cluster_raw[cluster_indices_list[i]]
                d_to_all_in_cluster = gower_distance_to_all_raw(
                    X_cluster_label, m,
                    feature_names=feature_names_raw,
                    symptom_cols=generic_symptoms_cols,
                    TEST_BOUNDS=TEST_BOUNDS,
                    symptom_range=(0, 10)
                )
                local_idx = int(np.argmin(d_to_all_in_cluster))
                global_idx = int(cluster_indices_list[i][local_idx])
                init_medoids.append(global_idx)

            k = len(init_medoids)
            f.write(f"K-Medoids clustering with k={k} (number of unique labels in neighbors)\n")

            init_matrix = D[init_medoids, :]

            kmedoids = KMedoids(
                n_clusters=k,
                metric="precomputed",
                init=init_matrix,
                max_iter=300,
                random_state=0
            )
            kmedoids.fit(D)

            medoids_idx = kmedoids.medoid_indices_
            cluster_labels_km = kmedoids.labels_
            inertia = kmedoids.inertia_
            f.write(f"cluster labels: {cluster_labels_km.tolist()}\n")
            f.write(f"Initial medoids indices: {init_medoids}\n")
            f.write(f"Final medoids indices: {medoids_idx.tolist()}\n\n")

            final_medoids_raw_df = X_cluster_raw_df.iloc[medoids_idx].copy().reset_index(drop=True)

            # Bin medoids to explainer space
            final_medoids_binned_df = bin_dataset(
                final_medoids_raw_df,
                TESTS=TESTS,
                generic_symptoms_cols=generic_symptoms_cols,
                verbose=False
            )
            # Overwrite generic symptoms with masked patient symptoms
            final_medoids_binned_df[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
            final_medoids_binned = final_medoids_binned_df.to_numpy()

            # -----------------------------
            # 7) BUILD EXPLAINERS
            # -----------------------------
            feature_cols = X_train.columns.tolist()
            categorical_names = {}
            indices = [i for i in range(len(subset_new_patient)) if i != idx_example_to_anchor]

            # Explainer for original & mean-instances modes
            explainer_orig = AnchorTabularExplainer(
                class_names=class_names,
                feature_names=feature_cols,
                train_data=subset_new_patient.iloc[indices].to_numpy(),
                discretizer=None,
                categorical_names=categorical_names,
            )

            # Explainer for medoid mode (uses full subset)
            explainer_medoid = AnchorTabularExplainer(
                class_names=class_names,
                feature_names=feature_cols,
                train_data=subset_new_patient.to_numpy(),
                discretizer=None,
                categorical_names=categorical_names,
            )

            # ----------------------------------------------------
            # 8) ORIGINAL MODE
            # ----------------------------------------------------
            f.write("ORIGINAL MODE\n")
            f.write(f"Anchor candidate index within subset: {idx_example_to_anchor}\n")
            f.write(f"True label of anchor instance: {y_subset_new_patient.iloc[idx_example_to_anchor]}\n")
            f.write(f"Predicted label of anchor instance (with hidden symptoms): "
                    f"{xgb_cl.predict(anchor_instance.reshape(1, -1))[0]}\n\n")

            exp_original, valid_anchors_original = explainer_orig.explain_instance(
                anchor_instance,
                xgb_cl,
                mode="conformal",
                query_label=label_to_exclude,
                qhat=qhat,
                threshold=0.95,
                delta=0.1,
                tau=0.15,
                beam_size=beam_size,
                predicate_mode="original",
                mean_instances=None
            )

            f.write("Main anchor: %s\n" % (' AND '.join(exp_original.names())))
            f.write("Precision: %.3f\n" % exp_original.precision())
            f.write("Coverage: %.5f\n" % exp_original.coverage())
            f.write("Cumulative coverage: %.5f\n" % exp_original.cumulative_coverage())

            # Predicate composition for MAIN anchor (original)
            main_pred_counts = count_predicate_types(exp_original.names())
            f.write(
                "Main anchor predicate composition (original): "
                f"generic={main_pred_counts['generic']}, "
                f"lung_tests={main_pred_counts['lung']}, "
                f"stomach_tests={main_pred_counts['stomach']}, "
                f"other={main_pred_counts['other']}\n"
            )


            main_names_orig = exp_original.names()
            main_applies_orig = anchor_applies_to_instance(main_names_orig, new_patient)
            f.write(f"Does MAIN anchor (original) apply to patient (with hidden symptoms)? "
                    f"{main_applies_orig}\n")

            num_valid_apply_orig = 0
            for va in valid_anchors_original:
                if anchor_applies_to_instance(va['names'], new_patient):
                    num_valid_apply_orig += 1
            f.write(
                f"#valid anchors (original) applying to patient: "
                f"{num_valid_apply_orig} / {len(valid_anchors_original)}\n"
            )

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
            for i_va, va in enumerate(sorted_valid_anchors_original, 1):
                names = " AND ".join(va['names'])
                prec = va['precision'][-1]
                cov = va['coverage'][-1]
                applies = anchor_applies_to_instance(va['names'], new_patient)
                pred_counts = count_predicate_types(va['names'])
                f.write(
                    f"  {i_va}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}, "
                    f"applies_to_patient={applies}\n"
                )
                f.write(
                    f"     predicate breakdown: "
                    f"generic={pred_counts['generic']}, "
                    f"lung_tests={pred_counts['lung']}, "
                    f"stomach_tests={pred_counts['stomach']}, "
                    f"other={pred_counts['other']}\n"
                )
            

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
            })

            # ----------------------------------------------------
            # 9) MEAN-INSTANCES MODE
            # ----------------------------------------------------
            f.write("MEAN-INSTANCES MODE\n")

            exp_mean, valid_anchors_mean = explainer_orig.explain_instance(
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

            f.write("Main anchor: %s\n" % (' AND '.join(exp_mean.names())))
            f.write("Precision: %.3f\n" % exp_mean.precision())
            f.write("Coverage: %.5f\n" % exp_mean.coverage())
            f.write("Cumulative coverage: %.5f\n" % exp_mean.cumulative_coverage())

            # Predicate composition for MAIN anchor (mean-instances)
            main_pred_counts = count_predicate_types(exp_mean.names())
            f.write(
                "Main anchor predicate composition (mean-instances): "
                f"generic={main_pred_counts['generic']}, "
                f"lung_tests={main_pred_counts['lung']}, "
                f"stomach_tests={main_pred_counts['stomach']}, "
                f"other={main_pred_counts['other']}\n"
            )


            main_names_mean = exp_mean.names()
            main_applies_mean = anchor_applies_to_instance(main_names_mean, new_patient)
            f.write(f"Does MAIN anchor (mean-instances) apply to patient (with hidden symptoms)? "
                    f"{main_applies_mean}\n")

            num_valid_apply_mean = 0
            for va in valid_anchors_mean:
                if anchor_applies_to_instance(va['names'], new_patient):
                    num_valid_apply_mean += 1
            f.write(
                f"#valid anchors (mean-instances) applying to patient: "
                f"{num_valid_apply_mean} / {len(valid_anchors_mean)}\n"
            )

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
            for i_va, va in enumerate(sorted_valid_anchors_mean, 1):
                names = " AND ".join(va['names'])
                prec = va['precision'][-1]
                cov = va['coverage'][-1]
                applies = anchor_applies_to_instance(va['names'], new_patient)
                f.write(
                    f"  {i_va}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}, "
                    f"applies_to_patient={applies}\n"
                )
                pred_counts = count_predicate_types(va['names'])
                f.write(
                    f"     predicate breakdown: "
                    f"generic={pred_counts['generic']}, "
                    f"lung_tests={pred_counts['lung']}, "
                    f"stomach_tests={pred_counts['stomach']}, "
                    f"other={pred_counts['other']}\n"
                )

            f.write("\n")
            # Collect all anchors in this mode: main + valid
            all_anchor_names_mean = [exp_mean.names()] + [va['names'] for va in valid_anchors_mean]
            total_anchors_mean = len(all_anchor_names_mean)

            num_generic_anchors_mean = 0
            num_lung_anchors_mean = 0
            num_stomach_anchors_mean = 0

            for names in all_anchor_names_mean:
                anchor_type = classify_anchor_type(names)
                if anchor_type == 'generic':
                    num_generic_anchors_mean += 1
                elif anchor_type == 'lung':
                    num_lung_anchors_mean += 1
                elif anchor_type == 'stomach':
                    num_stomach_anchors_mean += 1

            f.write("\nAnchor-type summary (MEAN-INSTANCES mode):\n")
            f.write(
                f"  generic-only anchors: {num_generic_anchors_mean} / {total_anchors_mean}\n"
            )
            f.write(
                f"  lung-only anchors: {num_lung_anchors_mean} / {total_anchors_mean}\n"
            )
            f.write(
                f"  stomach-only anchors: {num_stomach_anchors_mean} / {total_anchors_mean}\n"
            )
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
            })

            # ----------------------------------------------------
            # 10) MEDOID MODE (per-medoid anchors)
            # ----------------------------------------------------
            f.write("MEDOID MODE (KMedoids + RAW Gower)\n")

            all_medoid_anchors_for_instance = []

            for m_id, medoid_instance in enumerate(final_medoids_binned, 1):
                medoid_label = int(neighbors_labels.iloc[medoids_idx[m_id - 1]])

                exp_m, valid_anchors_m = explainer_medoid.explain_instance(
                    medoid_instance,
                    xgb_cl,
                    mode="conformal",
                    query_label=label_to_exclude,
                    qhat=qhat,
                    threshold=0.95,
                    delta=0.1,
                    tau=0.15,
                    beam_size=beam_size,
                    predicate_mode="medoid",
                    mean_instances=None
                )

                # Collect anchors for union pruning
                for va in valid_anchors_m:
                    all_medoid_anchors_for_instance.append({
                        'names': va['names'],
                        'medoid_id': m_id,
                        'source': 'valid_anchor',
                        'precision': va['precision'][-1],
                        'coverage': va['coverage'][-1],
                    })

                main_names_m = exp_m.names()
                already_present = any(
                    (a['names'] == main_names_m) and (a['medoid_id'] == m_id)
                    for a in all_medoid_anchors_for_instance
                )
                if not already_present:
                    all_medoid_anchors_for_instance.append({
                        'names': main_names_m,
                        'medoid_id': m_id,
                        'source': 'main_anchor',
                        'precision': exp_m.precision(),
                        'coverage': exp_m.coverage(),
                    })

                f.write(f"\n--- Medoid {m_id}/{len(final_medoids_binned)} ---\n")
                f.write(f"Medoid label: {medoid_label}\n")
                f.write("Main anchor: %s\n" % (' AND '.join(exp_m.names())))
                f.write("Precision: %.3f\n" % exp_m.precision())
                f.write("Coverage: %.5f\n" % exp_m.coverage())
                f.write("Cumulative coverage: %.5f\n" % exp_m.cumulative_coverage())

                # Predicate composition for MAIN anchor (medoid)
                main_pred_counts = count_predicate_types(exp_m.names())
                f.write(
                    "Main anchor predicate composition (medoid): "
                    f"generic={main_pred_counts['generic']}, "
                    f"lung_tests={main_pred_counts['lung']}, "
                    f"stomach_tests={main_pred_counts['stomach']}, "
                    f"other={main_pred_counts['other']}\n"
                )


                main_applies = anchor_applies_to_instance(exp_m.names(), new_patient)
                f.write(f"Does MAIN anchor (medoid {m_id}) apply to patient "
                        f"(with hidden symptoms)? {main_applies}\n")

                num_valid_apply = 0
                for va in valid_anchors_m:
                    if anchor_applies_to_instance(va['names'], new_patient):
                        num_valid_apply += 1
                f.write(
                    f"#valid anchors (medoid {m_id}) applying to patient: "
                    f"{num_valid_apply} / {len(valid_anchors_m)}\n"
                )

                if len(valid_anchors_m) > 0:
                    num_feats_each = [len(va['feature']) for va in valid_anchors_m]
                    avg_feats_m = float(np.mean(num_feats_each))
                    unique_feats_m = set()
                    for va in valid_anchors_m:
                        unique_feats_m |= set(va['feature'])
                    num_unique_feats_m = len(unique_feats_m)
                else:
                    avg_feats_m = 0.0
                    num_unique_feats_m = 0

                f.write(f"Valid anchors (medoid {m_id}): {len(valid_anchors_m)}\n")
                f.write(f"Avg #features per valid anchor (medoid {m_id}): {avg_feats_m:.3f}\n")
                f.write(f"#unique features across valid anchors (medoid {m_id}): "
                        f"{num_unique_feats_m}\n")

                sorted_valid = sorted(
                    valid_anchors_m,
                    key=lambda va: va['coverage'][-1],
                    reverse=True
                )
                for j, va in enumerate(sorted_valid, 1):
                    names = " AND ".join(va['names'])
                    prec = va['precision'][-1]
                    cov = va['coverage'][-1]
                    applies = anchor_applies_to_instance(va['names'], new_patient)
                    f.write(
                        f"  {j}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}, "
                        f"applies_to_patient={applies}\n"
                    )
                    pred_counts = count_predicate_types(va['names'])
                    f.write(
                        f"     predicate breakdown: "
                        f"generic={pred_counts['generic']}, "
                        f"lung_tests={pred_counts['lung']}, "
                        f"stomach_tests={pred_counts['stomach']}, "
                        f"other={pred_counts['other']}\n"
                    )

                f.write("\n")

                results.append({
                    'instance_idx': new_patient_idx,
                    'label_to_exclude': label_to_exclude,
                    'mode': 'medoid',
                    'medoid_id': m_id,
                    'main_precision': exp_m.precision(),
                    'main_coverage': exp_m.coverage(),
                    'cumulative_coverage': exp_m.cumulative_coverage(),
                    'avg_feats_valid': avg_feats_m,
                    'num_unique_feats_valid': num_unique_feats_m,
                    'num_valid_anchors': len(valid_anchors_m),
                    'main_applies': int(main_applies),
                    'num_valid_apply': num_valid_apply,
                })

            # ----------------------------------------------------
            # 11) UNION-PRUNED FINAL ANCHORS – MODE 1
            # ----------------------------------------------------
            f.write("\nUNION-PRUNED FINAL ANCHORS (across all medoids) – "
                    "MODE 1 (max marginal gain):\n")

            coverage_df = subset_new_patient
            flatten_tol = 1e-4

            final_anchors_union1, final_union_cov1, cov_traj_union1, gain_traj_union1 = (
                union_prune_anchors_mode1(
                    all_medoid_anchors_for_instance,
                    coverage_df,
                    n_samples=10000,
                    flatten_tol=flatten_tol
                )
            )

            f.write(f"Union coverage on coverage_df (MODE 1): {final_union_cov1:.5f}\n")
            f.write(f"Gain threshold used: {flatten_tol:.4f}\n")

            def extract_feature_name(cond_str):
                cond_str = cond_str.strip()
                for op in ["≤", "<=", ">", "="]:
                    if op in cond_str:
                        return cond_str.split(op, 1)[0].strip()
                return cond_str

            if len(final_anchors_union1) == 0:
                f.write("  No anchors selected by union pruning (MODE 1).\n\n")
            else:
                for idx_fa, fa in enumerate(final_anchors_union1, 1):
                    names_str = " AND ".join(fa['names'])
                    f.write(
                        f"  {idx_fa}) [medoid {fa['medoid_id']}, source={fa['source']}] "
                        f"{names_str} "
                        f"(cov_train={fa['cov_train']:.5f}, "
                        f"precision={fa['precision']:.3f}, "
                        f"coverage={fa['coverage']:.5f})\n"
                    )
                    # Predicate composition for this union anchor (MODE 1)
                    pred_counts = count_predicate_types(fa['names'])
                    f.write(
                        f"     predicate breakdown (union MODE 1): "
                        f"generic={pred_counts['generic']}, "
                        f"lung_tests={pred_counts['lung']}, "
                        f"stomach_tests={pred_counts['stomach']}, "
                        f"other={pred_counts['other']}\n"
                    )
                f.write("\n")
                lung_union1 = sum(
                    1 for a in final_anchors_union1
                    if a.get('disease_group') == "lung"
                )
                stomach_union1 = sum(
                    1 for a in final_anchors_union1
                    if a.get('disease_group') == "stomach"
                )
                f.write(
                    "Anchors (union MODE 1) by disease group for this label_to_exclude: "
                    f"lung={lung_union1}, stomach={stomach_union1}\n"
                )

                # Aggregated predicate counts over all union anchors (MODE 1)
                agg_union1 = {'generic': 0, 'lung': 0, 'stomach': 0, 'other': 0}
                for fa in final_anchors_union1:
                    c = count_predicate_types(fa['names'])
                    for k in agg_union1:
                        agg_union1[k] += c[k]

                f.write(
                    "AGGREGATED predicate counts (union MODE 1, this label_to_exclude): "
                    f"generic={agg_union1['generic']}, "
                    f"lung_tests={agg_union1['lung']}, "
                    f"stomach_tests={agg_union1['stomach']}, "
                    f"other={agg_union1['other']}\n\n"
                )

                total_union_anchors1 = len(final_anchors_union1)
                num_generic_union1 = 0
                num_lung_union1 = 0
                num_stomach_union1 = 0

                for fa in final_anchors_union1:
                    anchor_type = classify_anchor_type(fa['names'])
                    if anchor_type == 'generic':
                        num_generic_union1 += 1
                    elif anchor_type == 'lung':
                        num_lung_union1 += 1
                    elif anchor_type == 'stomach':
                        num_stomach_union1 += 1

                f.write("Anchor-type summary (UNION MODE 1):\n")
                f.write(
                    f"  generic-only anchors: {num_generic_union1} / {total_union_anchors1}\n"
                )
                f.write(
                    f"  lung-only anchors: {num_lung_union1} / {total_union_anchors1}\n"
                )
                f.write(
                    f"  stomach-only anchors: {num_stomach_union1} / {total_union_anchors1}\n"
                )
                f.write("\n")

                main_union_anchor1 = final_anchors_union1[0]
                main_prec_union1 = main_union_anchor1['precision']
                main_cov_union1 = main_union_anchor1['coverage']

                num_feats_each_union1 = [len(a['names']) for a in final_anchors_union1]
                avg_feats_union1 = float(np.mean(num_feats_each_union1))

                unique_feature_names_union1 = set()
                for a in final_anchors_union1:
                    for cond in a['names']:
                        unique_feature_names_union1.add(extract_feature_name(cond))
                num_unique_feats_union1 = len(unique_feature_names_union1)

                main_applies_union1 = int(
                    anchor_applies_to_instance(main_union_anchor1['names'], new_patient)
                )
                num_valid_apply_union1 = 0
                for a in final_anchors_union1:
                    if anchor_applies_to_instance(a['names'], new_patient):
                        num_valid_apply_union1 += 1

                f.write("UNION MODE 1 (per instance stats):\n")
                f.write(f"  #final anchors: {len(final_anchors_union1)}\n")
                f.write(f"  Main union-anchor precision: {main_prec_union1:.3f}\n")
                f.write(f"  Main union-anchor coverage (anchor's coverage): "
                        f"{main_cov_union1:.5f}\n")
                f.write(f"  Union coverage on coverage_df: {final_union_cov1:.5f}\n")
                f.write(f"  Avg #features per final anchor: {avg_feats_union1:.3f}\n")
                f.write(f"  #unique features across final anchors: {num_unique_feats_union1}\n")
                f.write(f"  Does MAIN union anchor apply to patient? "
                        f"{bool(main_applies_union1)}\n")
                f.write(
                    f"  #final anchors applying to patient: "
                    f"{num_valid_apply_union1} / {len(final_anchors_union1)}\n"
                )
                f.write("\n")

                results.append({
                    'instance_idx': new_patient_idx,
                    'label_to_exclude': label_to_exclude,
                    'mode': 'union_mode1',
                    'medoid_id': -1,
                    'main_precision': main_prec_union1,
                    'main_coverage': main_cov_union1,
                    'cumulative_coverage': final_union_cov1,
                    'avg_feats_valid': avg_feats_union1,
                    'num_unique_feats_valid': num_unique_feats_union1,
                    'num_valid_anchors': len(final_anchors_union1),
                    'main_applies': main_applies_union1,
                    'num_valid_apply': num_valid_apply_union1,
                    'coverage_traj': cov_traj_union1,
                    'gain_traj': gain_traj_union1,
                })

            # ----------------------------------------------------
            # 12) UNION-PRUNED FINAL ANCHORS – MODE 2
            # ----------------------------------------------------
            f.write("\nUNION-PRUNED FINAL ANCHORS (across all medoids) – "
                    "MODE 2 (sorted by individual coverage):\n")

            final_anchors_union2, final_union_cov2, cov_traj_union2, gain_traj_union2 = (
                union_prune_anchors_mode2(
                    all_medoid_anchors_for_instance,
                    coverage_df,
                    n_samples=10000,
                    flatten_tol=flatten_tol
                )
            )

            f.write(f"Union coverage on coverage_df (MODE 2): {final_union_cov2:.5f}\n")
            f.write(f"Gain threshold used: {flatten_tol:.4f}\n")

            if len(final_anchors_union2) == 0:
                f.write("  No anchors selected by union pruning (MODE 2).\n\n")
            else:
                for idx_fa, fa in enumerate(final_anchors_union2, 1):
                    names_str = " AND ".join(fa['names'])
                    f.write(
                        f"  {idx_fa}) [medoid {fa['medoid_id']}, source={fa['source']}] "
                        f"{names_str} "
                        f"(cov_train={fa['cov_train']:.5f}, "
                        f"precision={fa['precision']:.3f}, "
                        f"coverage={fa['coverage']:.5f})\n"
                    )
                    # Predicate composition for this union anchor (MODE 2)
                    pred_counts = count_predicate_types(fa['names'])
                    f.write(
                        f"     predicate breakdown (union MODE 2): "
                        f"generic={pred_counts['generic']}, "
                        f"lung_tests={pred_counts['lung']}, "
                        f"stomach_tests={pred_counts['stomach']}, "
                        f"other={pred_counts['other']}\n"
                    )

                f.write("\n")

                lung_union2 = sum(
                    1 for a in final_anchors_union2
                    if a.get('disease_group') == "lung"
                )
                stomach_union2 = sum(
                    1 for a in final_anchors_union2
                    if a.get('disease_group') == "stomach"
                )
                f.write(
                    "Anchors (union MODE 2) by disease group for this label_to_exclude: "
                    f"lung={lung_union2}, stomach={stomach_union2}\n"
                )

                # Aggregated predicate counts over all union anchors (MODE 2)
                agg_union2 = {'generic': 0, 'lung': 0, 'stomach': 0, 'other': 0}
                for fa in final_anchors_union2:
                    c = count_predicate_types(fa['names'])
                    for k in agg_union2:
                        agg_union2[k] += c[k]

                f.write(
                    "AGGREGATED predicate counts (union MODE 2, this label_to_exclude): "
                    f"generic={agg_union2['generic']}, "
                    f"lung_tests={agg_union2['lung']}, "
                    f"stomach_tests={agg_union2['stomach']}, "
                    f"other={agg_union2['other']}\n\n"
                )


                main_union_anchor2 = final_anchors_union2[0]
                main_prec_union2 = main_union_anchor2['precision']
                main_cov_union2 = main_union_anchor2['coverage']

                num_feats_each_union2 = [len(a['names']) for a in final_anchors_union2]
                avg_feats_union2 = float(np.mean(num_feats_each_union2))

                unique_feature_names_union2 = set()
                for a in final_anchors_union2:
                    for cond in a['names']:
                        unique_feature_names_union2.add(extract_feature_name(cond))
                num_unique_feats_union2 = len(unique_feature_names_union2)

                main_applies_union2 = int(
                    anchor_applies_to_instance(main_union_anchor2['names'], new_patient)
                )
                num_valid_apply_union2 = 0
                for a in final_anchors_union2:
                    if anchor_applies_to_instance(a['names'], new_patient):
                        num_valid_apply_union2 += 1

                f.write("UNION MODE 2 (per instance stats):\n")
                f.write(f"  #final anchors: {len(final_anchors_union2)}\n")
                f.write(f"  Main union-anchor precision: {main_prec_union2:.3f}\n")
                f.write(f"  Main union-anchor coverage (anchor's coverage): "
                        f"{main_cov_union2:.5f}\n")
                f.write(f"  Union coverage on coverage_df: {final_union_cov2:.5f}\n")
                f.write(f"  Avg #features per final anchor: {avg_feats_union2:.3f}\n")
                f.write(f"  #unique features across final anchors: {num_unique_feats_union2}\n")
                f.write(f"  Does MAIN union anchor apply to patient? "
                        f"{bool(main_applies_union2)}\n")
                f.write(
                    f"  #final anchors applying to patient: "
                    f"{num_valid_apply_union2} / {len(final_anchors_union2)}\n"
                )
                f.write("\n")

                results.append({
                    'instance_idx': new_patient_idx,
                    'label_to_exclude': label_to_exclude,
                    'mode': 'union_mode2',
                    'medoid_id': -1,
                    'main_precision': main_prec_union2,
                    'main_coverage': main_cov_union2,
                    'cumulative_coverage': final_union_cov2,
                    'avg_feats_valid': avg_feats_union2,
                    'num_unique_feats_valid': num_unique_feats_union2,
                    'num_valid_anchors': len(final_anchors_union2),
                    'main_applies': main_applies_union2,
                    'num_valid_apply': num_valid_apply_union2,
                    'coverage_traj': cov_traj_union2,
                    'gain_traj': gain_traj_union2,
                })

            f.write("-" * 80 + "\n\n")

    # ---------------------------------------------------------
    # 13) GLOBAL STATISTICS ACROSS ALL (INSTANCE, LABEL) PAIRS
    # ---------------------------------------------------------
    f.write("\n" + "=" * 100 + "\n")
    f.write("GLOBAL STATISTICS ACROSS ALL (INSTANCE, LABEL_TO_EXCLUDE) PAIRS\n")
    f.write("=" * 100 + "\n\n")

    for mode in ['original', 'mean', 'medoid', 'union_mode1', 'union_mode2']:
        mode_results = [r for r in results if r['mode'] == mode]
        if len(mode_results) == 0:
            continue

        cov_values = [r['main_coverage'] for r in mode_results]
        prec_values = [r['main_precision'] for r in mode_results]
        cum_cov_values = [r['cumulative_coverage'] for r in mode_results]
        feats_values = [r['avg_feats_valid'] for r in mode_results]
        unique_feats_values = [r['num_unique_feats_valid'] for r in mode_results]
        valid_anchors_values = [r['num_valid_anchors'] for r in mode_results]
        main_applies_values = [r.get('main_applies', 0) for r in mode_results]
        num_valid_apply_values = [r.get('num_valid_apply', 0) for r in mode_results]

        count_main_applies = int(np.sum(main_applies_values))
        frac_main_applies = count_main_applies / len(mode_results)

        count_at_least_one_valid = sum(1 for v in num_valid_apply_values if v > 0)
        frac_at_least_one_valid = count_at_least_one_valid / len(mode_results)

        avg_cov, std_cov = np.mean(cov_values), np.std(cov_values)
        avg_prec, std_prec = np.mean(prec_values), np.std(prec_values)
        avg_cum_cov, std_cum_cov = np.mean(cum_cov_values), np.std(cum_cov_values)
        avg_feats_valid, std_feats_valid = np.mean(feats_values), np.std(feats_values)
        avg_unique_feats_valid, std_unique_feats_valid = (
            np.mean(unique_feats_values), np.std(unique_feats_values)
        )
        avg_num_valid_anchors, std_num_valid_anchors = (
            np.mean(valid_anchors_values), np.std(valid_anchors_values)
        )

        f.write(f"MODE: {mode}\n")
        f.write(f"  Avg main precision: {avg_prec:.4f} (std={std_prec:.4f})\n")
        f.write(f"  Avg main coverage: {avg_cov:.4f} (std={std_cov:.4f})\n")
        f.write(f"  Avg cumulative coverage: {avg_cum_cov:.4f} (std={std_cum_cov:.4f})\n")
        f.write(f"  Avg #features per valid anchor: {avg_feats_valid:.4f} (std={std_feats_valid:.4f})\n")
        f.write(
            f"  Avg #unique features per instance (valid anchors): "
            f"{avg_unique_feats_valid:.4f} (std={std_unique_feats_valid:.4f})\n"
        )
        f.write(
            f"  Avg #valid anchors per instance: "
            f"{avg_num_valid_anchors:.4f} (std={std_num_valid_anchors:.4f})\n"
        )
        f.write(
            f"  #pairs where MAIN anchor applies: "
            f"{count_main_applies} / {len(mode_results)} "
            f"({frac_main_applies:.4f})\n"
        )
        f.write(
            f"  #pairs with ≥1 valid anchor applying: "
            f"{count_at_least_one_valid} / {len(mode_results)} "
            f"({frac_at_least_one_valid:.4f})\n"
        )
        f.write("\n")

    # ---------------------------------------------------------
    # 14) PLOTS OF UNION COVERAGE FLATTENING (MODES 1 & 2)
    # ---------------------------------------------------------
    for union_mode, suffix in [('union_mode1', 'mode1'), ('union_mode2', 'mode2')]:
        union_results = [r for r in results if r['mode'] == union_mode and 'coverage_traj' in r]

        if len(union_results) == 0:
            continue

        # Coverage trajectories
        max_len = max(len(r['coverage_traj']) for r in union_results)
        traj_matrix = np.full((len(union_results), max_len), np.nan, dtype=float)
        for i_r, r in enumerate(union_results):
            traj = r['coverage_traj']
            traj_matrix[i_r, :len(traj)] = traj

        mean_traj = np.nanmean(traj_matrix, axis=0)
        x = np.arange(1, max_len + 1)

        plt.figure()
        for i_r in range(traj_matrix.shape[0]):
            plt.plot(x, traj_matrix[i_r, :], alpha=0.2)
        plt.plot(x, mean_traj, linewidth=2)
        plt.xlabel("Number of union anchors added")
        plt.ylabel("Cumulative union coverage (sampled)")
        plt.title(f"Union coverage trajectories ({union_mode})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"union_coverage_flattens_{suffix}_20instances.png", dpi=200)
        plt.close()

        # Gain trajectories
        max_len_g = max(len(r['gain_traj']) for r in union_results)
        gain_matrix = np.full((len(union_results), max_len_g), np.nan, dtype=float)
        for i_r, r in enumerate(union_results):
            gtraj = r['gain_traj']
            gain_matrix[i_r, :len(gtraj)] = gtraj

        mean_gain = np.nanmean(gain_matrix, axis=0)
        xg = np.arange(1, max_len_g + 1)

        plt.figure()
        for i_r in range(gain_matrix.shape[0]):
            plt.plot(xg, gain_matrix[i_r, :], alpha=0.2)
        plt.plot(xg, mean_gain, linewidth=2)
        plt.xlabel("Anchor index in union selection order")
        plt.ylabel("Marginal gain in union coverage")
        plt.title(f"Marginal gains of union coverage ({union_mode})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"union_coverage_gains_{suffix}_20instances.png", dpi=200)
        plt.close()

    end_time = time.time()
    total_time = end_time - start_time
    f.write(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"\nDone. Full report saved to: {output_path}")
