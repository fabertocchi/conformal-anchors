# ---------------------------------------------------------
# IMPROVED EXPERIMENT WITH PROCEDURAL ANCHOR STATS:
# Evaluate all construction modes for excluding a label
# from the correct disease group (chosen at random)
# for up to 100 test instances.
#
# Modes:
#   - original
#   - mean-instances
#   - medoid (K-Medoids, RAW Gower)
#   - union mode 1 (max marginal gain)
#   - union mode 2 (sorted by individual coverage)
#
# For each mode, global statistics:
#   - avg main precision / coverage
#   - avg cumulative/union coverage
#   - avg #anchors per instance
#   - avg #unique features
#   - avg #features per anchor
#   - avg #procedural anchors per instance
#   - avg runtime per instance (where available)
#   - avg main procedural score
#   - fraction of main anchors that are procedural
#   - total procedural anchors / total anchors
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
# Generic symptom columns (binned space)
# ---------------------------------------------------------
generic_symptoms_cols = [
    'fever_severity', 'cough_severity', 'chest_pain_severity',
    'abdominal_pain_severity', 'fatigue_level', 'nausea'
]

# ---------------------------------------------------------
# FEATURE DOMAINS + PROCEDURAL SCORE
# ---------------------------------------------------------
# Domains from binned training data: distinct non-NaN values per feature
FEATURE_DOMAINS = {}
for col in X_train.columns:
    vals = sorted(X_train[col].dropna().unique())
    if len(vals) > 0:
        FEATURE_DOMAINS[col] = vals

# Threshold above which we consider an anchor "procedural"
# (i.e., each predicate covers a large portion of its feature's domain)
PROCEDURAL_THRESHOLD = 0.8


def parse_condition(cond):
    """
    Parse a condition string like:
      'pulmonary_function > 54.50'
      'chest_xray_score ≤ 0.00'
      'gastric_ph <= 3.00'
      'feature = 1.0'
    Returns: (feature_name, op, thresh_val, is_numeric)
      op in {'leq', 'gt', 'eq'} or None if cannot parse
    """
    cond = cond.strip()

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
        return None, None, None, False

    feature = feature.strip()
    thresh = thresh.strip()

    try:
        t_val = float(thresh)
        is_numeric = True
    except ValueError:
        t_val = thresh
        is_numeric = False

    return feature, op, t_val, is_numeric


def predicate_coverage_fraction(cond, feature_domains, eps=1e-20):
    """
    Given a single predicate string and FEATURE_DOMAINS,
    return the fraction of the feature's domain covered by this predicate.
    If cannot compute (e.g. feature not found or non-numeric), return None.
    """
    feature, op, t_val, is_numeric = parse_condition(cond)
    if feature is None or feature not in feature_domains:
        return None

    domain = feature_domains[feature]
    if len(domain) == 0:
        return None

    if is_numeric:
        domain_vals = np.array(domain, dtype=float)

        if op == "leq":
            covered = np.sum(domain_vals <= float(t_val))
        elif op == "gt":
            covered = np.sum(domain_vals > float(t_val))
        elif op == "eq":
            covered = np.sum(np.abs(domain_vals - float(t_val)) < eps)
        else:
            return None
    else:
        if op == "eq":
            covered = np.sum(np.array(domain, dtype=object) == t_val)
        else:
            return None

    return float(covered) / float(len(domain))


def anchor_procedural_score(predicate_names, feature_domains):
    """
    For an anchor (list of predicate strings), compute the mean
    fraction of the feature domain covered by each predicate.
    Return None if no predicate has a computable fraction.
    """
    fracs = []
    for cond in predicate_names:
        frac = predicate_coverage_fraction(cond, feature_domains)
        if frac is not None:
            fracs.append(frac)

    if not fracs:
        return None

    return float(np.mean(fracs))


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
            return False

        feature = feature.strip()
        thresh = thresh.strip()

        if feature not in patient_series.index:
            return False

        x = patient_series[feature]

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
# Union-pruning functions (Mode 1 & Mode 2)
# ---------------------------------------------------------
def union_prune_anchors_mode2(anchors, coverage_df, n_samples=10000, flatten_tol=1e-6):
    """
    Mode 2 (best anchors sorted by individual coverage).
    """
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


def union_prune_anchors_mode1(anchors, coverage_df, n_samples=10000, flatten_tol=1e-6):
    """
    Mode 1 (max marginal gain).
    Greedy selection by maximum marginal union coverage.
    """
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
# Load the trained model
# ---------------------------------------------------------
xgb_cl = xgb.XGBClassifier()
xgb_cl.load_model("xgb_model.json")
class_names = xgb_cl.classes_

start_time = time.time()

# ---------------------------------------------------------
# EXPERIMENT SETTINGS
# ---------------------------------------------------------
# k-NN subsets are precomputed once from full test set
all_subsets, y_subsets, similarities_list, indices_neighbors = get_knn_subsets(
    X_test, X_anchors, y_anchors, k=100
)

n_instances = 100
n_instances = min(n_instances, len(X_test))   # safety
beam_size = 10

selected_test_indices = np.random.choice(len(X_test), size=n_instances, replace=False)

results = []

output_path = "improved_anchor_experiment_results_full_framework_procedural.txt"

labels_stomach = [1, 5, 6, 7, 8]
labels_lung = [0, 2, 3, 4, 9]

# ---------------------------------------------------------
# MAIN LOOP OVER TEST INSTANCES
# ---------------------------------------------------------
with open(output_path, "w") as f:
    f.write("IMPROVED EXPERIMENT ON PROPOSED FRAMEWORK (all modes, with procedural stats)\n")
    f.write(f"Number of test instances: {n_instances}\n")
    f.write(f"Beam size: {beam_size}\n")
    f.write(f"Procedural threshold: {PROCEDURAL_THRESHOLD}\n")
    f.write("=" * 100 + "\n\n")

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

        subset_new_patient = all_subsets[new_patient_idx]
        y_subset_new_patient = y_subsets[new_patient_idx]

        subset_new_patient_raw = X_anchors_orig.iloc[indices_neighbors[new_patient_idx]].reset_index(drop=True)
        y_subset_new_patient_raw = y_anchors_orig.iloc[indices_neighbors[new_patient_idx]].reset_index(drop=True)

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
        # 3) CHOOSE CLASS TO EXCLUDE (SAME DISEASE GROUP)
        # -----------------------------
        if new_patient_true_label in labels_stomach:
            possible_labels = [i for i in labels_stomach if i != new_patient_true_label]
            label_to_exclude = np.random.choice(possible_labels)
        elif new_patient_true_label in labels_lung:
            possible_labels = [i for i in labels_lung if i != new_patient_true_label]
            label_to_exclude = np.random.choice(possible_labels)
        else:
            all_labels = list(class_names)
            possible_labels = [int(l) for l in all_labels if int(l) != int(new_patient_true_label)]
            label_to_exclude = np.random.choice(possible_labels)

        f.write(f"Class to exclude: {label_to_exclude} ({categories[label_to_exclude]})\n\n")

        pred_sets_names = [list(class_names[mask]) for mask in prediction_sets]

        pred_set_without_target = {}
        for i, pred_set in enumerate(pred_sets_names):
            if label_to_exclude not in pred_set:
                pred_set_without_target[i] = pred_set

        if len(pred_set_without_target) == 0:
            f.write("No prediction sets exclude the target label. Skipping this instance.\n\n")
            f.write("-" * 80 + "\n\n")
            continue

        f.write(f"{len(pred_set_without_target.keys())} prediction sets excluding class {label_to_exclude}:\n")
        f.write(str(pred_set_without_target) + "\n\n")

        idxs_neighbors_excluding_target = list(pred_set_without_target.keys())
        f.write(f"Neighbors excluding target indices: {idxs_neighbors_excluding_target}\n")
        neighbors_excluding_target = subset_new_patient.iloc[idxs_neighbors_excluding_target]
        neighbors_excluding_target_raw = subset_new_patient_raw.iloc[idxs_neighbors_excluding_target]
        neighbors_labels = y_subset_new_patient.iloc[idxs_neighbors_excluding_target]
        f.write(f"Neighbors excluding target labels: {neighbors_labels.tolist()}\n\n")

        # -----------------------------
        # 4) ORIGINAL MODE PREPARATION
        # -----------------------------
        idx_example_to_anchor = np.random.choice(list(pred_set_without_target.keys()))
        anchor_row = subset_new_patient.iloc[idx_example_to_anchor].copy()
        # Impute generic symptoms from actual patient
        anchor_row[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
        anchor_instance = anchor_row.to_numpy()

        # -----------------------------
        # 5) MEAN-INSTANCES & MEDOIDS (RAW GOWER)
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

        unique_labels = np.unique(neighbors_labels)
        f.write(f"Unique labels in neighbors excluding target: {unique_labels.tolist()}\n")

        mean_instances_raw = []
        cluster_list = []
        cluster_indices_list = []
        for label in unique_labels:
            mask = (neighbors_labels == label)
            cluster = neighbors_excluding_target_raw.loc[mask]
            cluster_indices = np.where(mask.values)[0]
            cluster_list.append(cluster)
            cluster_indices_list.append(cluster_indices)
            mean_instances_raw.append(cluster.mean(axis=0))
        mean_instances_raw = np.vstack(mean_instances_raw)

        # consistency checks
        for i, label in enumerate(unique_labels):
            assert (neighbors_labels.loc[cluster_list[i].index] == label).all()
        for i in range(len(unique_labels)):
            assert np.allclose(
                cluster_list[i][feature_names_raw].to_numpy(dtype=float),
                X_cluster_raw[cluster_indices_list[i]], equal_nan=True
            )

        # mean-instances (binned)
        mean_instances_raw_df = pd.DataFrame(mean_instances_raw, columns=feature_names_raw)
        mean_instances_binned_df = bin_dataset(
            mean_instances_raw_df,
            TESTS=TESTS,
            generic_symptoms_cols=generic_symptoms_cols,
            verbose=False
        )
        mean_instances_binned_df[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
        mean_instances_binned = mean_instances_binned_df.to_numpy()

        # K-Medoids init from means
        init_medoids = []
        for i, m in enumerate(mean_instances_raw):
            X_cluster = X_cluster_raw[cluster_indices_list[i]]
            d_to_all_in_cluster = gower_distance_to_all_raw(
                X_cluster, m,
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
        cluster_labels = kmedoids.labels_
        inertia = kmedoids.inertia_
        f.write(f"cluster labels: {cluster_labels.tolist()}\n")
        f.write(f"Initial medoids indices: {init_medoids}\n")
        f.write(f"Final medoids indices: {medoids_idx.tolist()}\n")

        final_medoids_raw_df = X_cluster_raw_df.iloc[medoids_idx].copy().reset_index(drop=True)

        final_medoids_binned_df = bin_dataset(
            final_medoids_raw_df,
            TESTS=TESTS,
            generic_symptoms_cols=generic_symptoms_cols,
            verbose=False
        )
        final_medoids_binned_df[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
        final_medoids_binned = final_medoids_binned_df.to_numpy()

        # -----------------------------
        # 6) BUILD EXPLAINERS
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

        explainer_mean = AnchorTabularExplainer(
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
        t0_mode = time.time()

        f.write(f"Anchor candidate index within subset: {idx_example_to_anchor}\n")
        f.write(f"True label of anchor instance: {y_subset_new_patient.iloc[idx_example_to_anchor]}\n")
        f.write(f"Predicted label of anchor instance: {xgb_cl.predict(anchor_instance.reshape(1, -1))[0]}\n\n")

        exp_original, valid_anchors_original = explainer_orig.explain_instance(
            anchor_instance, xgb_cl, mode="conformal",
            query_label=label_to_exclude, qhat=qhat,
            threshold=0.95, delta=0.1, tau=0.15, beam_size=beam_size,
        )

        runtime_orig = time.time() - t0_mode
        f.write(f"Runtime (original mode): {runtime_orig:.4f} seconds\n")

        f.write("Main anchor: %s\n" % (' AND '.join(exp_original.names())))
        f.write("Precision: %.3f\n" % exp_original.precision())
        f.write("Coverage: %.5f\n" % exp_original.coverage())
        f.write("Cumulative coverage: %.5f\n" % exp_original.cumulative_coverage())

        main_names_orig = exp_original.names()
        main_applies_orig = anchor_applies_to_instance(main_names_orig, new_patient)
        main_proc_score_orig = anchor_procedural_score(main_names_orig, FEATURE_DOMAINS)
        main_is_proc_orig = int(main_proc_score_orig is not None and main_proc_score_orig >= PROCEDURAL_THRESHOLD)

        f.write(f"Does MAIN anchor (original) apply to patient? {main_applies_orig}\n")
        f.write(
            "Procedural score (main, original): "
            f"{'%.3f' % main_proc_score_orig if main_proc_score_orig is not None else 'NA'}; "
            f"is_procedural={bool(main_is_proc_orig)}\n"
        )

        num_valid_apply_orig = 0
        num_proc_anchors_orig = 0
        for va in valid_anchors_original:
            if anchor_applies_to_instance(va['names'], new_patient):
                num_valid_apply_orig += 1
            score_va = anchor_procedural_score(va['names'], FEATURE_DOMAINS)
            if score_va is not None and score_va >= PROCEDURAL_THRESHOLD:
                num_proc_anchors_orig += 1

        f.write(
            f"#valid anchors (original) applying to patient: "
            f"{num_valid_apply_orig} / {len(valid_anchors_original)}\n"
        )
        f.write(
            f"#procedural anchors (original, using threshold {PROCEDURAL_THRESHOLD}): "
            f"{num_proc_anchors_orig} / {len(valid_anchors_original)}\n"
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

        for i, va in enumerate(sorted_valid_anchors_original, 1):
            names = " AND ".join(va['names'])
            prec = va['precision'][-1]
            cov = va['coverage'][-1]
            applies = anchor_applies_to_instance(va['names'], new_patient)
            score_va = anchor_procedural_score(va['names'], FEATURE_DOMAINS)
            is_proc_va = (score_va is not None and score_va >= PROCEDURAL_THRESHOLD)
            f.write(
                f"  {i}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}, "
                f"applies_to_patient={applies}\n"
            )
            f.write(
                f"     procedural_score={ '%.3f' % score_va if score_va is not None else 'NA' }, "
                f"is_procedural={is_proc_va}\n"
            )
        f.write("\n")

        results.append({
            'instance_idx': new_patient_idx,
            'mode': 'original',
            'main_precision': exp_original.precision(),
            'main_coverage': exp_original.coverage(),
            'cumulative_coverage': exp_original.cumulative_coverage(),
            'avg_feats_valid': avg_feats,
            'num_unique_feats_valid': num_unique_feats,
            'num_valid_anchors': len(valid_anchors_original),
            'num_procedural_anchors': num_proc_anchors_orig,
            'main_applies': int(main_applies_orig),
            'num_valid_apply': num_valid_apply_orig,
            'main_procedural_score': main_proc_score_orig,
            'main_is_procedural': main_is_proc_orig,
            'runtime': runtime_orig,
        })

        # ----------------------------------------------------
        # 7b) MEAN-INSTANCES MODE
        # ----------------------------------------------------
        f.write("MEAN-INSTANCES MODE\n")
        t0_mode = time.time()

        all_valid_anchors_mean = []
        # One call with mean_instances passed to explainer_mean (or explainer_orig).
        exp_mean, valid_anchors_mean = explainer_mean.explain_instance(
            anchor_instance,
            xgb_cl,
            mode="conformal",
            query_label=label_to_exclude,
            qhat=qhat,
            threshold=0.95,
            delta=0.1,
            tau=0.15,
            beam_size=beam_size,
            # assuming your AnchorTabularExplainer supports this:
            mean_instances=mean_instances_binned,
        )

        runtime_mean = time.time() - t0_mode
        f.write(f"Runtime (mean-instances mode): {runtime_mean:.4f} seconds\n")

        f.write("Main anchor: %s\n" % (' AND '.join(exp_mean.names())))
        f.write("Precision: %.3f\n" % exp_mean.precision())
        f.write("Coverage: %.5f\n" % exp_mean.coverage())
        f.write("Cumulative coverage: %.5f\n" % exp_mean.cumulative_coverage())

        main_names_mean = exp_mean.names()
        main_applies_mean = anchor_applies_to_instance(main_names_mean, new_patient)
        main_proc_score_mean = anchor_procedural_score(main_names_mean, FEATURE_DOMAINS)
        main_is_proc_mean = int(main_proc_score_mean is not None and main_proc_score_mean >= PROCEDURAL_THRESHOLD)

        f.write(f"Does MAIN anchor (mean-instances) apply to patient? {main_applies_mean}\n")
        f.write(
            "Procedural score (main, mean-instances): "
            f"{'%.3f' % main_proc_score_mean if main_proc_score_mean is not None else 'NA'}; "
            f"is_procedural={bool(main_is_proc_mean)}\n"
        )

        num_valid_apply_mean = 0
        num_proc_anchors_mean = 0
        if len(valid_anchors_mean) > 0:
            num_feats_each_m = [len(va['feature']) for va in valid_anchors_mean]
            avg_feats_mean = float(np.mean(num_feats_each_m))
            unique_feats_mean = set()
            for va in valid_anchors_mean:
                unique_feats_mean |= set(va['feature'])

                if anchor_applies_to_instance(va['names'], new_patient):
                    num_valid_apply_mean += 1
                score_va = anchor_procedural_score(va['names'], FEATURE_DOMAINS)
                if score_va is not None and score_va >= PROCEDURAL_THRESHOLD:
                    num_proc_anchors_mean += 1

            num_unique_feats_mean = len(unique_feats_mean)
        else:
            avg_feats_mean = 0.0
            num_unique_feats_mean = 0

        f.write(
            f"#valid anchors (mean-instances) applying to patient: "
            f"{num_valid_apply_mean} / {len(valid_anchors_mean)}\n"
        )
        f.write(
            f"#procedural anchors (mean-instances): "
            f"{num_proc_anchors_mean} / {len(valid_anchors_mean)}\n"
        )

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
            score_va = anchor_procedural_score(va['names'], FEATURE_DOMAINS)
            is_proc_va = (score_va is not None and score_va >= PROCEDURAL_THRESHOLD)
            f.write(
                f"  {i}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}, "
                f"applies_to_patient={applies}\n"
            )
            f.write(
                f"     procedural_score={ '%.3f' % score_va if score_va is not None else 'NA' }, "
                f"is_procedural={is_proc_va}\n"
            )
        f.write("\n")

        results.append({
            'instance_idx': new_patient_idx,
            'mode': 'mean',
            'main_precision': exp_mean.precision(),
            'main_coverage': exp_mean.coverage(),
            'cumulative_coverage': exp_mean.cumulative_coverage(),
            'avg_feats_valid': avg_feats_mean,
            'num_unique_feats_valid': num_unique_feats_mean,
            'num_valid_anchors': len(valid_anchors_mean),
            'num_procedural_anchors': num_proc_anchors_mean,
            'main_applies': int(main_applies_mean),
            'num_valid_apply': num_valid_apply_mean,
            'main_procedural_score': main_proc_score_mean,
            'main_is_procedural': main_is_proc_mean,
            'runtime': runtime_mean,
        })

        # ----------------------------------------------------
        # 8) MEDOID MODE (KMedoids + RAW Gower)
        # ----------------------------------------------------
        f.write("MEDOID MODE (KMedoids + RAW Gower)\n")
        t0_medoid_mode = time.time()

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
                beam_size=7
            )

            # collect medoid anchors for union modes
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

            main_applies = anchor_applies_to_instance(exp_m.names(), new_patient)
            main_proc_score_med = anchor_procedural_score(exp_m.names(), FEATURE_DOMAINS)
            main_is_proc_med = int(main_proc_score_med is not None and main_proc_score_med >= PROCEDURAL_THRESHOLD)
            f.write(f"Does MAIN anchor (medoid {m_id}) apply to patient? {main_applies}\n")
            f.write(
                "Procedural score (main, medoid %d): %s; is_procedural=%s\n"
                % (
                    m_id,
                    '%.3f' % main_proc_score_med if main_proc_score_med is not None else 'NA',
                    str(bool(main_is_proc_med)),
                )
            )

            num_valid_apply = 0
            num_proc_anchors_med = 0
            if len(valid_anchors_m) > 0:
                num_feats_each_m = [len(va['feature']) for va in valid_anchors_m]
                avg_feats_m = float(np.mean(num_feats_each_m))
                unique_feats_m = set()
                for va in valid_anchors_m:
                    unique_feats_m |= set(va['feature'])

                    if anchor_applies_to_instance(va['names'], new_patient):
                        num_valid_apply += 1
                    score_va = anchor_procedural_score(va['names'], FEATURE_DOMAINS)
                    if score_va is not None and score_va >= PROCEDURAL_THRESHOLD:
                        num_proc_anchors_med += 1

                num_unique_feats_m = len(unique_feats_m)
            else:
                avg_feats_m = 0.0
                num_unique_feats_m = 0

            f.write(
                f"#valid anchors (medoid {m_id}) applying to patient: "
                f"{num_valid_apply} / {len(valid_anchors_m)}\n"
            )
            f.write(
                f"#procedural anchors (medoid {m_id}): "
                f"{num_proc_anchors_med} / {len(valid_anchors_m)}\n"
            )
            f.write(f"Valid anchors (medoid {m_id}): {len(valid_anchors_m)}\n")
            f.write(f"Avg #features per valid anchor (medoid {m_id}): {avg_feats_m:.3f}\n")
            f.write(f"#unique features across valid anchors (medoid {m_id}): {num_unique_feats_m}\n")

            sorted_valid = sorted(valid_anchors_m, key=lambda va: va['coverage'][-1], reverse=True)
            for j, va in enumerate(sorted_valid, 1):
                names = " AND ".join(va['names'])
                prec = va['precision'][-1]
                cov = va['coverage'][-1]
                applies = anchor_applies_to_instance(va['names'], new_patient)
                score_va = anchor_procedural_score(va['names'], FEATURE_DOMAINS)
                is_proc_va = (score_va is not None and score_va >= PROCEDURAL_THRESHOLD)
                f.write(
                    f"  {j}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}, "
                    f"applies_to_patient={applies}\n"
                )
                f.write(
                    f"     procedural_score={ '%.3f' % score_va if score_va is not None else 'NA' }, "
                    f"is_procedural={is_proc_va}\n"
                )

            results.append({
                'instance_idx': new_patient_idx,
                'mode': 'medoid',
                'medoid_id': m_id,
                'main_precision': exp_m.precision(),
                'main_coverage': exp_m.coverage(),
                'cumulative_coverage': exp_m.cumulative_coverage(),
                'avg_feats_valid': avg_feats_m,
                'num_unique_feats_valid': num_unique_feats_m,
                'num_valid_anchors': len(valid_anchors_m),
                'num_procedural_anchors': num_proc_anchors_med,
                'main_applies': int(main_applies),
                'num_valid_apply': num_valid_apply,
                'main_procedural_score': main_proc_score_med,
                'main_is_procedural': main_is_proc_med,
                'runtime': None,  # medoid runtime aggregated later
            })
            f.write("\n")

        runtime_medoid = time.time() - t0_medoid_mode
        f.write(f"Total runtime (medoid mode, all medoids): {runtime_medoid:.4f} seconds\n\n")

        # ----------------------------------------------------
        # 8b) UNION-PRUNED FINAL ANCHORS (modes 1 & 2)
        # ----------------------------------------------------
        coverage_df = subset_new_patient
        flatten_tol = 1e-4

        def extract_feature_name(cond_str):
            cond_str = cond_str.strip()
            for op in ["≤", "<=", ">", "="]:
                if op in cond_str:
                    return cond_str.split(op, 1)[0].strip()
            return cond_str

        # UNION MODE 1
        f.write("\nUNION MODE 1 (max marginal gain):\n")
        t0_union1 = time.time()
        final_anchors_u1, final_union_cov_u1, cov_traj_u1, gain_traj_u1 = union_prune_anchors_mode1(
            all_medoid_anchors_for_instance,
            coverage_df,
            n_samples=10000,
            flatten_tol=flatten_tol
        )
        runtime_u1 = time.time() - t0_union1
        f.write(f"Union coverage (mode 1): {final_union_cov_u1:.5f}\n")
        f.write(f"Runtime (union mode 1): {runtime_u1:.4f} seconds\n")

        num_proc_u1 = 0
        if len(final_anchors_u1) == 0:
            f.write("  No anchors selected by union mode 1.\n\n")
        else:
            for idx_fa, fa in enumerate(final_anchors_u1, 1):
                names_str = " AND ".join(fa['names'])
                score_fa = anchor_procedural_score(fa['names'], FEATURE_DOMAINS)
                is_proc_fa = (score_fa is not None and score_fa >= PROCEDURAL_THRESHOLD)
                if is_proc_fa:
                    num_proc_u1 += 1

                f.write(
                    f"  {idx_fa}) [medoid {fa['medoid_id']}, source={fa['source']}] "
                    f"{names_str} "
                    f"(cov_train={fa['cov_train']:.5f}, "
                    f"precision={fa['precision']:.3f}, "
                    f"coverage={fa['coverage']:.5f})\n"
                )
                f.write(
                    f"     procedural_score={ '%.3f' % score_fa if score_fa is not None else 'NA' }, "
                    f"is_procedural={is_proc_fa}\n"
                )
            f.write(
                f"#procedural anchors (union mode 1): "
                f"{num_proc_u1} / {len(final_anchors_u1)}\n\n"
            )

            main_union_anchor_u1 = final_anchors_u1[0]
            main_prec_u1 = main_union_anchor_u1['precision']
            main_cov_u1 = main_union_anchor_u1['coverage']
            main_proc_score_u1 = anchor_procedural_score(main_union_anchor_u1['names'], FEATURE_DOMAINS)
            main_is_proc_u1 = int(main_proc_score_u1 is not None and main_proc_score_u1 >= PROCEDURAL_THRESHOLD)

            num_feats_each_u1 = [len(a['names']) for a in final_anchors_u1]
            avg_feats_u1 = float(np.mean(num_feats_each_u1))

            unique_feature_names_u1 = set()
            for a in final_anchors_u1:
                for cond in a['names']:
                    unique_feature_names_u1.add(extract_feature_name(cond))
            num_unique_feats_u1 = len(unique_feature_names_u1)

            main_applies_u1 = int(anchor_applies_to_instance(main_union_anchor_u1['names'], new_patient))
            num_valid_apply_u1 = 0
            for a in final_anchors_u1:
                if anchor_applies_to_instance(a['names'], new_patient):
                    num_valid_apply_u1 += 1

            f.write("UNION MODE 1 (per instance stats):\n")
            f.write(f"  #final anchors: {len(final_anchors_u1)}\n")
            f.write(f"  Main union-anchor precision: {main_prec_u1:.3f}\n")
            f.write(f"  Main union-anchor coverage: {main_cov_u1:.5f}\n")
            f.write(f"  Union coverage on coverage_df: {final_union_cov_u1:.5f}\n")
            f.write(f"  Avg #features per final anchor: {avg_feats_u1:.3f}\n")
            f.write(f"  #unique features across final anchors: {num_unique_feats_u1}\n")
            f.write(
                f"  Does MAIN union anchor apply to patient? {bool(main_applies_u1)}\n"
            )
            f.write(
                f"  #final anchors applying to patient: "
                f"{num_valid_apply_u1} / {len(final_anchors_u1)}\n"
            )
            f.write(
                "  Main union-anchor procedural score: "
                f"{'%.3f' % main_proc_score_u1 if main_proc_score_u1 is not None else 'NA'}; "
                f"is_procedural={bool(main_is_proc_u1)}\n\n"
            )

            results.append({
                'instance_idx': new_patient_idx,
                'mode': 'union1',
                'medoid_id': -1,
                'main_precision': main_prec_u1,
                'main_coverage': main_cov_u1,
                'cumulative_coverage': final_union_cov_u1,
                'avg_feats_valid': avg_feats_u1,
                'num_unique_feats_valid': num_unique_feats_u1,
                'num_valid_anchors': len(final_anchors_u1),
                'num_procedural_anchors': num_proc_u1,
                'main_applies': main_applies_u1,
                'num_valid_apply': num_valid_apply_u1,
                'coverage_traj': cov_traj_u1,
                'gain_traj': gain_traj_u1,
                'main_procedural_score': main_proc_score_u1,
                'main_is_procedural': main_is_proc_u1,
                'runtime': runtime_u1,
            })

        # UNION MODE 2
        f.write("\nUNION MODE 2 (best anchors by coverage):\n")
        t0_union2 = time.time()
        final_anchors_u2, final_union_cov_u2, cov_traj_u2, gain_traj_u2 = union_prune_anchors_mode2(
            all_medoid_anchors_for_instance,
            coverage_df,
            n_samples=10000,
            flatten_tol=flatten_tol
        )
        runtime_u2 = time.time() - t0_union2
        f.write(f"Union coverage (mode 2): {final_union_cov_u2:.5f}\n")
        f.write(f"Runtime (union mode 2): {runtime_u2:.4f} seconds\n")

        num_proc_u2 = 0
        if len(final_anchors_u2) == 0:
            f.write("  No anchors selected by union mode 2.\n\n")
        else:
            for idx_fa, fa in enumerate(final_anchors_u2, 1):
                names_str = " AND ".join(fa['names'])
                score_fa = anchor_procedural_score(fa['names'], FEATURE_DOMAINS)
                is_proc_fa = (score_fa is not None and score_fa >= PROCEDURAL_THRESHOLD)
                if is_proc_fa:
                    num_proc_u2 += 1

                f.write(
                    f"  {idx_fa}) [medoid {fa['medoid_id']}, source={fa['source']}] "
                    f"{names_str} "
                    f"(cov_train={fa['cov_train']:.5f}, "
                    f"precision={fa['precision']:.3f}, "
                    f"coverage={fa['coverage']:.5f})\n"
                )
                f.write(
                    f"     procedural_score={ '%.3f' % score_fa if score_fa is not None else 'NA' }, "
                    f"is_procedural={is_proc_fa}\n"
                )
            f.write(
                f"#procedural anchors (union mode 2): "
                f"{num_proc_u2} / {len(final_anchors_u2)}\n\n"
            )

            main_union_anchor_u2 = final_anchors_u2[0]
            main_prec_u2 = main_union_anchor_u2['precision']
            main_cov_u2 = main_union_anchor_u2['coverage']
            main_proc_score_u2 = anchor_procedural_score(main_union_anchor_u2['names'], FEATURE_DOMAINS)
            main_is_proc_u2 = int(main_proc_score_u2 is not None and main_proc_score_u2 >= PROCEDURAL_THRESHOLD)

            num_feats_each_u2 = [len(a['names']) for a in final_anchors_u2]
            avg_feats_u2 = float(np.mean(num_feats_each_u2))

            unique_feature_names_u2 = set()
            for a in final_anchors_u2:
                for cond in a['names']:
                    unique_feature_names_u2.add(extract_feature_name(cond))
            num_unique_feats_u2 = len(unique_feature_names_u2)

            main_applies_u2 = int(anchor_applies_to_instance(main_union_anchor_u2['names'], new_patient))
            num_valid_apply_u2 = 0
            for a in final_anchors_u2:
                if anchor_applies_to_instance(a['names'], new_patient):
                    num_valid_apply_u2 += 1

            f.write("UNION MODE 2 (per instance stats):\n")
            f.write(f"  #final anchors: {len(final_anchors_u2)}\n")
            f.write(f"  Main union-anchor precision: {main_prec_u2:.3f}\n")
            f.write(f"  Main union-anchor coverage: {main_cov_u2:.5f}\n")
            f.write(f"  Union coverage on coverage_df: {final_union_cov_u2:.5f}\n")
            f.write(f"  Avg #features per final anchor: {avg_feats_u2:.3f}\n")
            f.write(f"  #unique features across final anchors: {num_unique_feats_u2}\n")
            f.write(
                f"  Does MAIN union anchor apply to patient? {bool(main_applies_u2)}\n"
            )
            f.write(
                f"  #final anchors applying to patient: "
                f"{num_valid_apply_u2} / {len(final_anchors_u2)}\n"
            )
            f.write(
                "  Main union-anchor procedural score: "
                f"{'%.3f' % main_proc_score_u2 if main_proc_score_u2 is not None else 'NA'}; "
                f"is_procedural={bool(main_is_proc_u2)}\n\n"
            )

            results.append({
                'instance_idx': new_patient_idx,
                'mode': 'union2',
                'medoid_id': -1,
                'main_precision': main_prec_u2,
                'main_coverage': main_cov_u2,
                'cumulative_coverage': final_union_cov_u2,
                'avg_feats_valid': avg_feats_u2,
                'num_unique_feats_valid': num_unique_feats_u2,
                'num_valid_anchors': len(final_anchors_u2),
                'num_procedural_anchors': num_proc_u2,
                'main_applies': main_applies_u2,
                'num_valid_apply': num_valid_apply_u2,
                'coverage_traj': cov_traj_u2,
                'gain_traj': gain_traj_u2,
                'main_procedural_score': main_proc_score_u2,
                'main_is_procedural': main_is_proc_u2,
                'runtime': runtime_u2,
            })

        f.write("-" * 80 + "\n\n")

    # ---------------------------------------------------------
    # 9) GLOBAL STATISTICS ACROSS ALL INSTANCES
    # ---------------------------------------------------------
    f.write("\n" + "=" * 100 + "\n")
    f.write("GLOBAL STATISTICS ACROSS ALL CONSIDERED INSTANCES\n")
    f.write("=" * 100 + "\n\n")

    for mode in ['original', 'mean', 'medoid', 'union1', 'union2']:
        mode_results = [r for r in results if r['mode'] == mode]
        if len(mode_results) == 0:
            continue

        cov_values = [r['main_coverage'] for r in mode_results]
        prec_values = [r['main_precision'] for r in mode_results]
        cum_cov_values = [r['cumulative_coverage'] for r in mode_results]
        feats_values = [r['avg_feats_valid'] for r in mode_results]
        unique_feats_values = [r['num_unique_feats_valid'] for r in mode_results]
        valid_anchors_values = [r['num_valid_anchors'] for r in mode_results]
        procedural_anchors_values = [r.get('num_procedural_anchors', 0) for r in mode_results]
        main_applies_values = [r['main_applies'] for r in mode_results]
        num_valid_apply_values = [r['num_valid_apply'] for r in mode_results]
        runtime_values = [r['runtime'] for r in mode_results if r.get('runtime') is not None]

        main_proc_scores = [r.get('main_procedural_score', None) for r in mode_results]
        # convert to numpy array with NaNs
        main_proc_scores_clean = np.array(
            [s if s is not None else np.nan for s in main_proc_scores],
            dtype=float
        )
        main_is_proc_values = [r.get('main_is_procedural', 0) for r in mode_results]

        count_main_applies = int(np.sum(main_applies_values))
        frac_main_applies = count_main_applies / len(mode_results)

        count_at_least_one_valid = sum(1 for v in num_valid_apply_values if v > 0)
        frac_at_least_one_valid = count_at_least_one_valid / len(mode_results)

        avg_cov, std_cov = np.mean(cov_values), np.std(cov_values)
        avg_prec, std_prec = np.mean(prec_values), np.std(prec_values)
        avg_cum_cov, std_cum_cov = np.mean(cum_cov_values), np.std(cum_cov_values)
        avg_feats_valid, std_feats_valid = np.mean(feats_values), np.std(feats_values)
        avg_unique_feats_valid, std_unique_feats_valid = np.mean(unique_feats_values), np.std(unique_feats_values)
        avg_num_valid_anchors, std_num_valid_anchors = np.mean(valid_anchors_values), np.std(valid_anchors_values)
        avg_proc_anchors, std_proc_anchors = np.mean(procedural_anchors_values), np.std(procedural_anchors_values)

        total_anchors_mode = int(np.sum(valid_anchors_values))
        total_proc_anchors_mode = int(np.sum(procedural_anchors_values))
        frac_proc_overall = (
            total_proc_anchors_mode / total_anchors_mode
            if total_anchors_mode > 0 else 0.0
        )

        # main procedural stats
        frac_main_procedural = np.sum(main_is_proc_values) / len(mode_results)
        avg_main_proc_score = float(np.nanmean(main_proc_scores_clean))

        f.write(f"MODE: {mode}\n")
        f.write(f"  Avg main precision: {avg_prec:.4f} (std={std_prec:.4f})\n")
        f.write(f"  Avg main coverage: {avg_cov:.4f} (std={std_cov:.4f})\n")
        f.write(f"  Avg cumulative/union coverage: {avg_cum_cov:.4f} (std={std_cum_cov:.4f})\n")
        f.write(f"  Avg #features per anchor: {avg_feats_valid:.4f} (std={std_feats_valid:.4f})\n")
        f.write(f"  Avg #unique features per instance: {avg_unique_feats_valid:.4f} (std={std_unique_feats_valid:.4f})\n")
        f.write(f"  Avg #anchors per instance: {avg_num_valid_anchors:.4f} (std={std_num_valid_anchors:.4f})\n")
        f.write(f"  Avg #procedural anchors per instance: {avg_proc_anchors:.4f} (std={std_proc_anchors:.4f})\n")
        f.write(
            f"  Total procedural anchors: {total_proc_anchors_mode} / {total_anchors_mode} "
            f"({frac_proc_overall:.4f})\n"
        )
        f.write(
            f"  #instances where MAIN anchor applies: "
            f"{count_main_applies} / {len(mode_results)} "
            f"({frac_main_applies:.4f})\n"
        )
        f.write(
            f"  #instances with ≥1 anchor applying: "
            f"{count_at_least_one_valid} / {len(mode_results)} "
            f"({frac_at_least_one_valid:.4f})\n"
        )
        f.write(
            f"  Fraction of MAIN anchors that are procedural: "
            f"{frac_main_procedural:.4f}\n"
        )
        f.write(
            f"  Avg main procedural score (ignoring NaNs): "
            f"{avg_main_proc_score:.4f}\n"
        )

        if runtime_values:
            avg_runtime, std_runtime = np.mean(runtime_values), np.std(runtime_values)
            f.write(f"  Avg runtime per instance (seconds): {avg_runtime:.4f} (std={std_runtime:.4f})\n")

        if mode == 'medoid':
            medoid_counts = {}
            for r in mode_results:
                inst = r['instance_idx']
                m_id = r.get('medoid_id', None)
                if m_id is None:
                    continue
                if inst not in medoid_counts:
                    medoid_counts[inst] = set()
                medoid_counts[inst].add(m_id)

            if medoid_counts:
                counts = [len(s) for s in medoid_counts.values()]
                avg_medoids = np.mean(counts)
                std_medoids = np.std(counts)
                min_medoids = int(np.min(counts))
                max_medoids = int(np.max(counts))

                f.write(
                    f"  Avg #medoids per instance: {avg_medoids:.4f} "
                    f"(std={std_medoids:.4f}, min={min_medoids}, max={max_medoids})\n"
                )

        f.write("\n")

    # ---------------------------------------------------------
    # 10) PLOTS OF UNION COVERAGE FLATTENING (MODES 1 & 2)
    # ---------------------------------------------------------
    for mode, prefix in [('union1', 'mode1'), ('union2', 'mode2')]:
        union_results = [r for r in results if r['mode'] == mode and 'coverage_traj' in r]

        if len(union_results) == 0:
            continue

        max_len = max(len(r['coverage_traj']) for r in union_results)
        traj_matrix = np.full((len(union_results), max_len), np.nan, dtype=float)
        for i, r in enumerate(union_results):
            traj = r['coverage_traj']
            traj_matrix[i, :len(traj)] = traj

        mean_traj = np.nanmean(traj_matrix, axis=0)
        x = np.arange(1, max_len + 1)

        plt.figure()
        for i in range(traj_matrix.shape[0]):
            plt.plot(x, traj_matrix[i, :], alpha=0.2)
        plt.plot(x, mean_traj, linewidth=2)
        plt.xlabel("Number of union anchors added")
        plt.ylabel("Cumulative union coverage (sampled)")
        plt.title(f"Union coverage trajectories ({mode})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"union_coverage_flattens_{prefix}_10_bestanchors.png", dpi=200)
        plt.close()

        max_len_g = max(len(r['gain_traj']) for r in union_results)
        gain_matrix = np.full((len(union_results), max_len_g), np.nan, dtype=float)
        for i, r in enumerate(union_results):
            gtraj = r['gain_traj']
            gain_matrix[i, :len(gtraj)] = gtraj

        mean_gain = np.nanmean(gain_matrix, axis=0)
        xg = np.arange(1, max_len_g + 1)

        plt.figure()
        for i in range(gain_matrix.shape[0]):
            plt.plot(xg, gain_matrix[i, :], alpha=0.2)
        plt.plot(xg, mean_gain, linewidth=2)
        plt.xlabel("Anchor index in union selection order")
        plt.ylabel("Marginal gain in union coverage")
        plt.title(f"Marginal gains of union coverage ({mode})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"union_coverage_gains_{prefix}_10_bestanchors.png", dpi=200)
        plt.close()

    end_time = time.time()
    total_time = end_time - start_time
    f.write(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")

print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"\nDone. Full report saved to: {output_path}")