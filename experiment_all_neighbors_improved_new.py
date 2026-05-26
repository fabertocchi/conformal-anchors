# ---------------------------------------------------------
# IMPROVED Experiment: Exclude TRUE label
# - original vs K-Medoids (RAW Gower) vs UNION pruning
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
from subset import get_knn_subsets, gower_similarity_mixed_raw
from prediction_set import compute_conformal_prediction_set_batch
import matplotlib.pyplot as plt
from data.tests_v2 import TESTS
from binning import bin_dataset
from dataset_basic_setting import TEST_BOUNDS
from sklearn_extra.cluster import KMedoids
import pandas as pd
import numpy as np
import time

np.random.seed(1)

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

        # Remove spaces
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
            if not (x <= t_val):
                return False
        elif op == "gt":
            if not numeric:
                return False
            if not (x > t_val):
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

    # All predicates satisfied
    return True


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
    print(D)
    return D

def union_prune_anchors(anchors, coverage_df, n_samples=10000, flatten_tol=1e-6):
    """
    Mode 1
    Union-pruning over a collection of anchors for a single test instance, using
    coverage estimated on n_samples points drawn WITH REPLACEMENT from
    coverage_df (which should be subset_new_patient).

    At each step, among all remaining anchors, we pick the one that yields
    the largest increase in UNION coverage. We stop when the best possible
    gain is <= flatten_tol.

    Parameters
    ----------
    anchors : list of dict
        Each dict MUST contain at least:
          - 'names': list of predicate-strings for the anchor.

    coverage_df : pandas.DataFrame
        Universe to sample from (e.g. subset_new_patient).

    n_samples : int, default=10000
        Number of samples drawn with replacement from coverage_df.

    flatten_tol : float, default=1e-6
        Minimum required increase in cumulative coverage (on the sampled data)
        to keep adding anchors. If the best possible gain is <= flatten_tol,
        we stop.

    Returns
    -------
    final_anchors : list of dict
        Subset of anchors selected by the union-pruning strategy.
        Each anchor dict is augmented with:
          - 'coverage_idx': set of sample indices it covers
          - 'cov_train': float, individual coverage on the sampled data

    final_union_coverage : float
        Cumulative coverage of the union of all selected anchors on the
        sampled data.
    """
    if not anchors:
        return [], 0.0

    n_universe = coverage_df.shape[0]
    if n_universe == 0:
        return [], 0.0

    # ----------------------------------------------------
    # 1) SAMPLE EXACTLY n_samples ROWS WITH REPLACEMENT
    #    from coverage_df (subset_new_patient)
    # ----------------------------------------------------
    sampled_idx = np.random.choice(range(n_universe), size=n_samples, replace=True)
    coverage_data = coverage_df.iloc[sampled_idx].reset_index(drop=True)
    n_cov = coverage_data.shape[0]   # should be n_samples

    # ----------------------------------------------------
    # 2) PRECOMPUTE COVERAGE INDICES PER ANCHOR
    # ----------------------------------------------------
    state = {'t_coverage_idx': {}}

    for t, a in enumerate(anchors):
        # For each row in coverage_data, check if anchor applies
        mask = coverage_data.apply(
            lambda row: anchor_applies_to_instance(a['names'], row),
            axis=1
        ).to_numpy(dtype=bool)

        covered_idx = set(np.where(mask)[0])

        # Save indices and individual coverage on sampled data
        state['t_coverage_idx'][t] = covered_idx
        a['coverage_idx'] = covered_idx
        a['cov_train'] = float(len(covered_idx)) / n_cov

    # ----------------------------------------------------
    # 3) GREEDY UNION SELECTION BY MAX MARGINAL GAIN
    # ----------------------------------------------------
    final_anchors = []
    covered_union = set()
    last_cumulative_coverage = 0.0

    remaining = set(range(len(anchors)))  # anchor indices not yet selected
    rank = 0
    coverage_trajectory = []   # cumulative coverage after each accepted anchor
    gain_trajectory = []       # marginal gain for each accepted anchor


    while remaining:
        best_t = None
        best_gain = 0.0
        best_candidate_union = None

        # Search anchor with maximum marginal gain in union coverage
        for t in remaining:
            candidate_union = covered_union | state['t_coverage_idx'][t]
            cumulative_coverage = float(len(candidate_union)) / n_cov
            gain = cumulative_coverage - last_cumulative_coverage

            if gain > best_gain:
                best_gain = gain
                best_t = t
                best_candidate_union = candidate_union

        # If even the best possible gain is not above tolerance, stop
        if best_t is None or best_gain <= flatten_tol:
            break

        # Otherwise, accept that anchor
        rank += 1
        final_anchors.append(anchors[best_t])
        covered_union = best_candidate_union
        last_cumulative_coverage = float(len(covered_union)) / n_cov
        remaining.remove(best_t)

        # Record trajectories
        coverage_trajectory.append(last_cumulative_coverage)
        gain_trajectory.append(best_gain)

    return final_anchors, last_cumulative_coverage, coverage_trajectory, gain_trajectory

# def union_prune_anchors(anchors, coverage_df, n_samples=10000, flatten_tol=1e-6):
#     """
#     Mode 2
#     Union-pruning over a collection of anchors for a single test instance, using
#     coverage estimated on n_samples points drawn WITH REPLACEMENT from
#     coverage_df (which should be subset_new_patient).

#     Returns
#     -------
#     final_anchors : list of dict
#     final_union_coverage : float
#     coverage_trajectory : list of float
#         Cumulative coverage after each accepted anchor.
#     gain_trajectory : list of float
#         Marginal gain in coverage for each accepted anchor.
#     """
#     if not anchors:
#         return [], 0.0, [], []

#     n_universe = coverage_df.shape[0]
#     if n_universe == 0:
#         return [], 0.0, [], []

#     # 1) SAMPLE EXACTLY n_samples ROWS WITH REPLACEMENT
#     sampled_idx = np.random.choice(range(n_universe), size=n_samples, replace=True)
#     coverage_data = coverage_df.iloc[sampled_idx].reset_index(drop=True)
#     n_cov = coverage_data.shape[0]   # will be n_samples

#     # 2) PRECOMPUTE COVERAGE INDICES PER ANCHOR
#     state = {'t_coverage_idx': {}}

#     for t, a in enumerate(anchors):
#         mask = coverage_data.apply(
#             lambda row: anchor_applies_to_instance(a['names'], row),
#             axis=1
#         ).to_numpy(dtype=bool)

#         covered_idx = set(np.where(mask)[0])

#         state['t_coverage_idx'][t] = covered_idx
#         a['coverage_idx'] = covered_idx
#         a['cov_train'] = float(len(covered_idx)) / n_cov

#     # 3) SORT ANCHORS BY INDIVIDUAL COVERAGE (DESCENDING)
#     sorted_t = sorted(range(len(anchors)), key=lambda t: anchors[t]['cov_train'], reverse=True)

#     # 4) GREEDY UNION SELECTION UNTIL CUMULATIVE COVERAGE FLATTENS
#     final_anchors = []
#     covered_union = set()
#     last_cumulative_coverage = 0.0

#     coverage_trajectory = []   # cumulative coverage after each accepted anchor
#     gain_trajectory = []       # marginal gain for each accepted anchor

#     for rank, t in enumerate(sorted_t, start=1):
#         candidate_union = covered_union | state['t_coverage_idx'][t]
#         cumulative_coverage = float(len(candidate_union)) / n_cov

#         if rank == 1:
#             # always take the highest-coverage anchor
#             final_anchors.append(anchors[t])
#             covered_union = candidate_union
#             last_cumulative_coverage = cumulative_coverage

#             coverage_trajectory.append(cumulative_coverage)
#             gain_trajectory.append(cumulative_coverage)  # gain from 0
#         else:
#             gain = cumulative_coverage - last_cumulative_coverage

#             if gain <= flatten_tol:
#                 # curve is "flat" -> stop
#                 break

#             final_anchors.append(anchors[t])
#             covered_union = candidate_union
#             last_cumulative_coverage = cumulative_coverage

#             coverage_trajectory.append(cumulative_coverage)
#             gain_trajectory.append(gain)

#     return final_anchors, last_cumulative_coverage, coverage_trajectory, gain_trajectory


# ---------------------------------------------------------
# 0) Setup
# ---------------------------------------------------------
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

n_instances = 100
n_instances = min(n_instances, len(X_test))   # safety
beam_size = 1

# Choose which test indices to use (here: random without replacement)
selected_test_indices = np.random.choice(len(X_test), size=n_instances, replace=False)
# selected_test_indices = [506, 10963, 10442, 6503, 7449, 1186]
# To store per-(instance,mode) stats
results = []

output_path = "improved_anchor_experiment_TRUE_label_exclusion_all_neighbors_1_new_applicability_new.txt"

with open(output_path, "w") as f:
    f.write("IMPROVED EXPERIMENT: Exclude TRUE label (original vs K-Medoids vs UNION)\n")
    f.write(f"Number of test instances: {n_instances}\n")
    f.write(f"Beam size: {beam_size}\n")
    f.write("=" * 100 + "\n\n")

    # Count instances where NO neighbor prediction set excludes the TRUE label
    counter_no_ps_excluding_true = 0

    for run_id, new_patient_idx in enumerate(selected_test_indices, 1):  # start counting from 1
        
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
        f.write(f"Prediction sets shape: {prediction_sets.shape}\n")

        # -----------------------------
        # 3) CHOOSE CLASS TO EXCLUDE = TRUE LABEL
        # -----------------------------
        label_to_exclude = int(new_patient_true_label)
        f.write(f"Class to exclude (TRUE label): {label_to_exclude} ({categories[label_to_exclude]})\n\n")

        pred_sets_names = [list(class_names[mask]) for mask in prediction_sets]

        pred_set_without_target = {}
        for i, pred_set in enumerate(pred_sets_names):
            if label_to_exclude not in pred_set:
                pred_set_without_target[i] = pred_set

        if len(pred_set_without_target) == 0:
            f.write("No prediction sets exclude the TRUE label. Skipping this instance.\n\n")
            f.write("-" * 80 + "\n\n")
            counter_no_ps_excluding_true += 1
            continue

        f.write(f"{len(pred_set_without_target.keys())} prediction sets excluding TRUE label {label_to_exclude}:\n")
        f.write(str(pred_set_without_target) + "\n\n")

        # Extract neighbors excluding target label and their labels
        idxs_neighbors_excluding_target = list(pred_set_without_target.keys())
        f.write(f"Neighbors excluding TRUE label indices: {idxs_neighbors_excluding_target}\n")
        neighbors_excluding_target = subset_new_patient.iloc[idxs_neighbors_excluding_target]
        neighbors_excluding_target_raw = subset_new_patient_raw.iloc[idxs_neighbors_excluding_target]
        neighbors_labels = y_subset_new_patient.iloc[idxs_neighbors_excluding_target]
        f.write(f"Neighbors excluding TRUE label labels: {neighbors_labels.tolist()}\n\n")

        # -----------------------------
        # 4) ORIGINAL MODE PREPARATION
        # -----------------------------
        # Choose anchor instance within subset
        idx_example_to_anchor = np.random.choice(list(pred_set_without_target.keys()))
        anchor_row = subset_new_patient.iloc[idx_example_to_anchor].copy()
        # Impute generic symptoms from the actual patient
        anchor_row[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
        anchor_instance = anchor_row.to_numpy()

        # -----------------------------
        # 5) K-MEDOIDS MODE PREPARATION (with RAW GOWER)
        # -----------------------------
        # Raw neighbors to cluster (DataFrame -> keep column names aligned)
        X_cluster_raw_df = neighbors_excluding_target_raw.reset_index(drop=True)
        X_cluster_raw = X_cluster_raw_df.to_numpy(dtype=float)
        feature_names_raw = X_cluster_raw_df.columns.tolist()

        # Precompute distance matrix among neighbors using RAW Gower
        D = compute_gower_distance_matrix_raw(
            X_cluster_raw,
            feature_names=feature_names_raw,
            symptom_cols=generic_symptoms_cols,
            TEST_BOUNDS=TEST_BOUNDS,
            symptom_range=(0, 10)
        )

        # Build raw mean-instances per label
        unique_labels = np.unique(neighbors_labels)
        f.write(f"Unique labels in neighbors excluding TRUE label: {unique_labels.tolist()}\n")
        mean_instances_raw = []
        cluster_list = []
        cluster_indices_list = []
        for label in unique_labels:
            # For each label take all neighbors with that label
            mask = (neighbors_labels == label)
            # Cluster as a DataFrame (keeps original index from neighbors_excluding_target_raw)
            cluster = neighbors_excluding_target_raw.loc[mask]
            # Positional indices where mask is True
            cluster_indices = np.where(mask.values)[0]
            cluster_list.append(cluster)
            cluster_indices_list.append(cluster_indices)
            # Compute feature-wise mean
            mean_instances_raw.append(cluster.mean(axis=0))
        mean_instances_raw = np.vstack(mean_instances_raw)  # each row is a mean-instance
        
        for i, label in enumerate(unique_labels):
            assert (neighbors_labels.loc[cluster_list[i].index] == label).all()
            
        for i in range(len(unique_labels)):
            assert np.allclose(
                cluster_list[i][feature_names_raw].to_numpy(dtype=float),
                X_cluster_raw[cluster_indices_list[i]], equal_nan=True
            )

        # Snap each mean to nearest actual point => init medoids (must be actual data points)
        init_medoids = []
        used = set()        # To enforce unique initial medoids
        for i, m in enumerate(mean_instances_raw):
            X_cluster = X_cluster_raw[cluster_indices_list[i]]
            # Compute distance of each per-label mean to all points
            d_to_all_in_cluster = gower_distance_to_all_raw(
                X_cluster, m,
                feature_names=feature_names_raw,
                symptom_cols=generic_symptoms_cols,
                TEST_BOUNDS=TEST_BOUNDS,
                symptom_range=(0, 10)
            )
            # Sort distances and pick closest real point not already used (first candidate that isn't already taken)
            local_idx = int(np.argmin(d_to_all_in_cluster))
            global_idx = int(cluster_indices_list[i][local_idx])

            init_medoids.append(global_idx)

        # Run KMedoids (sklearn-extra) on PRECOMPUTED distance matrix
        k = len(init_medoids)
        f.write(f"K-Medoids clustering with k={k} (number of unique labels in neighbors)\n")
        init_matrix = D[init_medoids, :]   # shape (k, n_pts)

        kmedoids = KMedoids(
            n_clusters=k,
            metric="precomputed",
            init=init_matrix,
            max_iter=300,
            random_state=0
        )
        kmedoids.fit(D)

        medoids_idx = kmedoids.medoid_indices_          # Indices of final selected medoids
        cluster_labels = kmedoids.labels_               # Cluster assignment for each neighbor
        inertia = kmedoids.inertia_                     # Sum of distances of points to their medoids
        f.write(f"cluster labels: {cluster_labels.tolist()}\n")
        f.write(f"Initial medoids indices: {init_medoids}\n")
        f.write(f"Final medoids indices: {medoids_idx.tolist()}\n")
        # Final medoids in RAW space (actual neighbor rows)
        final_medoids_raw_df = X_cluster_raw_df.iloc[medoids_idx].copy().reset_index(drop=True)

        # Bin medoids so they match the explainer feature space (subset_new_patient is binned)
        final_medoids_binned_df = bin_dataset(
            final_medoids_raw_df,
            TESTS=TESTS,
            generic_symptoms_cols=generic_symptoms_cols,
            verbose=False
        )

        # Force generic symptoms to patient values
        final_medoids_binned_df[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
        final_medoids_binned = final_medoids_binned_df.to_numpy()

        # -----------------------------
        # 6) BUILD EXPLAINER (once per instance)
        # -----------------------------
        feature_cols = X_train.columns.tolist()
        categorical_names = {}
        indices = [i for i in range(len(subset_new_patient)) if i != idx_example_to_anchor]

        explainer_orig = AnchorTabularExplainer(
            class_names=class_names,     
            feature_names=feature_cols,
            train_data=subset_new_patient.iloc[indices].to_numpy(),                     
            discretizer=None,
            categorical_names=categorical_names,
        )

        explainer_medoid = AnchorTabularExplainer(
            class_names=class_names,     
            feature_names=feature_cols,
            train_data=subset_new_patient.to_numpy(),                     
            discretizer=None,
            categorical_names=categorical_names,
        )

        # ----------------------------------------------------
        # 7) ORIGINAL MODE
        # ----------------------------------------------------
        f.write("ORIGINAL MODE\n")
        f.write(f"Anchor candidate index within subset: {idx_example_to_anchor}\n")
        f.write(f"True label of anchor instance: {y_subset_new_patient.iloc[idx_example_to_anchor]}\n")
        f.write(f"Predicted label of anchor instance: {xgb_cl.predict(anchor_instance.reshape(1, -1))[0]}\n\n")

        exp_original, valid_anchors_original = explainer_orig.explain_instance(
            anchor_instance, xgb_cl, mode="conformal",
            query_label=label_to_exclude, qhat=qhat, 
            threshold=0.95, delta=0.1, tau=0.15, beam_size=beam_size,
        )

        f.write("Main anchor: %s\n" % (' AND '.join(exp_original.names())))
        f.write("Precision: %.3f\n" % exp_original.precision())
        f.write("Coverage: %.5f\n" % exp_original.coverage())
        f.write("Cumulative coverage: %.5f\n" % exp_original.cumulative_coverage())

        # Check if main anchor applies to the patient
        main_names_orig = exp_original.names()
        main_applies_orig = anchor_applies_to_instance(main_names_orig, new_patient)
        f.write(f"Does MAIN anchor (original) apply to patient? {main_applies_orig}\n")

        # Check how many valid anchors apply to the patient
        num_valid_apply_orig = 0
        for va in valid_anchors_original:
            if anchor_applies_to_instance(va['names'], new_patient):
                num_valid_apply_orig += 1
        f.write(
            f"#valid anchors (original) applying to patient: "
            f"{num_valid_apply_orig} / {len(valid_anchors_original)}\n"
        )

        # Per-instance statistics on valid anchors
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

        # Print valid anchors sorted by coverage
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
            f.write(
                f"  {i}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}, "
                f"applies_to_patient={applies}\n"
            )
        f.write("\n")

        # Store results for global statistics
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
            # same definition as in the TRUE-label-ALL-neighbors experiment
            'no_anchor_applies': int((not main_applies_orig) and (num_valid_apply_orig == 0)),
        })


        # ----------------------------------------------------
        # 8) K-MEDOID MODE
        # ----------------------------------------------------
        f.write("MEDOID MODE (KMedoids + RAW Gower)\n")

        # Collect all medoid anchors for this instance (for union pruning)
        all_medoid_anchors_for_instance = []

        for m_id, medoid_instance in enumerate(final_medoids_binned, 1):
            # Label of the medoid point (in neighbors_excluding_target)
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
                beam_size=beam_size
            )

            # --- Collect anchors from this medoid for later union pruning ---
            # 1) Add all valid anchors with their precision/coverage
            for va in valid_anchors_m:
                all_medoid_anchors_for_instance.append({
                    'names': va['names'],              # list of predicate strings
                    'medoid_id': m_id,
                    'source': 'valid_anchor',
                    'precision': va['precision'][-1],
                    'coverage': va['coverage'][-1],
                })

            # 2) Ensure the MAIN anchor is also included (if not already)
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

            # Check whether main anchor applies to the patient
            main_applies = anchor_applies_to_instance(exp_m.names(), new_patient)
            f.write(f"Does MAIN anchor (medoid {m_id}) apply to patient? {main_applies}\n")

            # Check how many valid anchors apply to the patient
            num_valid_apply = 0
            for va in valid_anchors_m:
                if anchor_applies_to_instance(va['names'], new_patient):
                    num_valid_apply += 1
            f.write(
                f"#valid anchors (medoid {m_id}) applying to patient: "
                f"{num_valid_apply} / {len(valid_anchors_m)}\n"
            )

            # Compute complexity statistics on valid anchors for this medoid
            if len(valid_anchors_m) > 0:
                num_feats_each = [len(va['feature']) for va in valid_anchors_m]
                avg_feats = float(np.mean(num_feats_each))
                unique_feats = set()
                for va in valid_anchors_m:
                    unique_feats |= set(va['feature'])
                num_unique_feats = len(unique_feats)
            else:
                avg_feats = 0.0
                num_unique_feats = 0

            f.write(f"Valid anchors (medoid {m_id}): {len(valid_anchors_m)}\n")
            f.write(f"Avg #features per valid anchor (medoid {m_id}): {avg_feats:.3f}\n")
            f.write(f"#unique features across valid anchors (medoid {m_id}): {num_unique_feats}\n")

            # Print all valid anchors sorted by coverage
            sorted_valid = sorted(valid_anchors_m, key=lambda va: va['coverage'][-1], reverse=True)
            for j, va in enumerate(sorted_valid, 1):
                names = " AND ".join(va['names'])
                prec = va['precision'][-1]
                cov = va['coverage'][-1]
                applies = anchor_applies_to_instance(va['names'], new_patient)
                f.write(
                    f"  {j}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}, "
                    f"applies_to_patient={applies}\n"
                )

            # Store results 
            results.append({
                'instance_idx': new_patient_idx,
                'label_to_exclude': label_to_exclude,
                'mode': 'medoid',
                'medoid_id': m_id,  
                'main_precision': exp_m.precision(),
                'main_coverage': exp_m.coverage(),
                'cumulative_coverage': exp_m.cumulative_coverage(),
                'avg_feats_valid': avg_feats,
                'num_unique_feats_valid': num_unique_feats,
                'num_valid_anchors': len(valid_anchors_m),
                'main_applies': int(main_applies),
                'num_valid_apply': num_valid_apply,
                'no_anchor_applies': int((not main_applies) and (num_valid_apply == 0)),
            })

            f.write("\n")

        # ----------------------------------------------------
        # 8b) UNION-PRUNED FINAL ANCHORS (ACROSS ALL MEDOIDS)
        # ----------------------------------------------------
        f.write("\nUNION-PRUNED FINAL ANCHORS (across all medoids):\n")

        # coverage_df is the same universe as explainer.train_data
        coverage_df = subset_new_patient
        flatten_tol = 1e-4

        final_anchors_union, final_union_cov, coverage_trajectory, gain_trajectory = union_prune_anchors(
            all_medoid_anchors_for_instance,
            coverage_df,
            n_samples=10000,
            flatten_tol=flatten_tol
        )

        f.write(f"Union coverage on coverage_df (train_data universe): "
                f"{final_union_cov:.5f}\n")
        f.write(f"Flatten tolerance used: {flatten_tol:.4f}\n")

        if len(final_anchors_union) == 0:
            f.write("  No anchors selected by union pruning.\n\n")
        else:
            # Log final union-pruned anchors
            for idx_fa, fa in enumerate(final_anchors_union, 1):
                names_str = " AND ".join(fa['names'])
                f.write(
                    f"  {idx_fa}) [medoid {fa['medoid_id']}, source={fa['source']}] "
                    f"{names_str} "
                    f"(cov_train={fa['cov_train']:.5f}, "
                    f"precision={fa['precision']:.3f}, "
                    f"coverage={fa['coverage']:.5f})\n"
                )

            f.write("\n")

            # Treat the first selected anchor as the "main" union anchor
            main_union_anchor = final_anchors_union[0]
            main_prec_union = main_union_anchor['precision']
            main_cov_union = main_union_anchor['coverage']

            # #features in each anchor (use number of predicates in 'names')
            num_feats_each_union = [len(a['names']) for a in final_anchors_union]
            avg_feats_union = float(np.mean(num_feats_each_union))

            # #unique feature names across all final anchors (from predicates)
            def extract_feature_name(cond_str):
                cond_str = cond_str.strip()
                for op in ["≤", "<=", ">", "="]:
                    if op in cond_str:
                        return cond_str.split(op, 1)[0].strip()
                return cond_str  # fallback: whole string

            unique_feature_names_union = set()
            for a in final_anchors_union:
                for cond in a['names']:
                    unique_feature_names_union.add(extract_feature_name(cond))
            num_unique_feats_union = len(unique_feature_names_union)

            # How many union anchors apply to the CURRENT PATIENT?
            main_applies_union = int(anchor_applies_to_instance(main_union_anchor['names'], new_patient))
            num_valid_apply_union = 0
            for a in final_anchors_union:
                if anchor_applies_to_instance(a['names'], new_patient):
                    num_valid_apply_union += 1

            # Log union stats for this instance
            f.write("UNION MODE (per instance stats):\n")
            f.write(f"  #final anchors: {len(final_anchors_union)}\n")
            f.write(f"  Main union-anchor precision: {main_prec_union:.3f}\n")
            f.write(f"  Main union-anchor coverage (anchor's coverage): {main_cov_union:.5f}\n")
            f.write(f"  Union coverage on coverage_df: {final_union_cov:.5f}\n")
            f.write(f"  Avg #features per final anchor: {avg_feats_union:.3f}\n")
            f.write(f"  #unique features across final anchors: {num_unique_feats_union}\n")
            f.write(f"  Does MAIN union anchor apply to patient? {bool(main_applies_union)}\n")
            f.write(
                f"  #final anchors applying to patient: "
                f"{num_valid_apply_union} / {len(final_anchors_union)}\n"
            )
            f.write("\n")

            # Store union stats in results (mode='union')
            results.append({
                'instance_idx': new_patient_idx,
                'label_to_exclude': label_to_exclude,
                'mode': 'union',
                'medoid_id': -1,  # not tied to a single medoid
                'main_precision': main_prec_union,
                'main_coverage': main_cov_union,
                'cumulative_coverage': final_union_cov,          # union coverage
                'avg_feats_valid': avg_feats_union,
                'num_unique_feats_valid': num_unique_feats_union,
                'num_valid_anchors': len(final_anchors_union),
                'main_applies': main_applies_union,
                'num_valid_apply': num_valid_apply_union,
                'no_anchor_applies': int((not main_applies_union) and (num_valid_apply_union == 0)),
            })


    # ---------------------------------------------------------
    # 9) GLOBAL STATISTICS ACROSS ALL INSTANCES
    # ---------------------------------------------------------
    f.write("\n" + "=" * 100 + "\n")
    f.write("GLOBAL STATISTICS ACROSS ALL (INSTANCE, TRUE_LABEL_EXCLUSION) PAIRS\n")
    f.write("=" * 100 + "\n\n")

    f.write(
        f"Number of instances with no neighbors' prediction sets excluding TRUE label: "
        f"{counter_no_ps_excluding_true}\n\n"
    )

    for mode in ['original', 'medoid', 'union']:
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
