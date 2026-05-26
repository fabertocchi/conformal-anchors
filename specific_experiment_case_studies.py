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

HEALTHY_RANGES = {
    'pulmonary_function': 85,
    'sputum_neutrophil_percent': 29.5,
    'wbc_count': 7500.5,
    'hemoglobin': 13.05,       # sex-agnostic mid-normal range
    'gastric_ph': 1.5,
    'chest_xray_score': 0,       # clear / minimal findings
    'endoscopy_score': 0,
    'h_pylori_level': 0,
}

N_SAMPLES_PROCEDURAL = 10000      # number of bootstrap samples from neighbors
DELTA_PROCEDURAL_THRESHOLD = 0.02 # how close precisions must be to call it procedural

# def anchor_applies_to_instance(predicate_names, patient_series, use_healthy_for_nan=False):
#     """
#     predicate_names: list of strings like
#         "pulmonary_function > 54.50", "chest_xray_score ≤ 0.00", ...
#     patient_series: pandas Series, features indexed by column name.

#     use_healthy_for_nan:
#         - If False (default): if x is NaN, the condition is NOT satisfied (old behavior).
#         - If True: if x is NaN *and* the feature is in HEALTHY_RANGES, a healthy
#           value is sampled in the given range and used in the comparison.

#     Returns True if ALL predicates are satisfied by the (possibly imputed) values.
#     """
#     for cond in predicate_names:
#         cond = cond.strip()

#         # Determine operator and split
#         if "≤" in cond:
#             feature, thresh = cond.split("≤", 1)
#             op = "leq"
#         elif "<=" in cond:
#             feature, thresh = cond.split("<=", 1)
#             op = "leq"
#         elif ">" in cond:
#             feature, thresh = cond.split(">", 1)
#             op = "gt"
#         elif "=" in cond:
#             feature, thresh = cond.split("=", 1)
#             op = "eq"
#         else:
#             # Unknown pattern -> be conservative: treat as not satisfied
#             return False

#         feature = feature.strip()
#         thresh = thresh.strip()

#         if feature not in patient_series.index:
#             return False

#         x = patient_series[feature]

#         # Handle missing values
#         if pd.isna(x):
#             # If requested and we know a healthy range for this feature,
#             # sample a healthy value and continue.
#             if use_healthy_for_nan and feature in HEALTHY_RANGES:
#                 x = HEALTHY_RANGES[feature]  
#                 # low, high = HEALTHY_RANGES[feature]
#                 # x = (high+low)/2
#             else:
#                 # Old behavior: condition not satisfied
#                 return False

#         # Try to interpret threshold as numeric
#         try:
#             t_val = float(thresh)
#             numeric = True
#         except ValueError:
#             numeric = False

#         if op == "leq":
#             if not numeric or not (x <= t_val):
#                 return False
#         elif op == "gt":
#             if not numeric or not (x > t_val):
#                 return False
#         elif op == "eq":
#             if numeric:
#                 if not (abs(x - t_val) < 1e-20):
#                     return False
#             else:
#                 if str(x) != thresh:
#                     return False
#         else:
#             return False

#     return True

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


def extract_feature_name(cond_str):
    """
    Extract the feature name from a predicate like:
      'pulmonary_function > 54.50'
      'chest_xray_score ≤ 0.00'
      'gastric_ph <= 3.00'
      'hemoglobin = 13.05'
    """
    cond_str = cond_str.strip()
    for op in ["≤", "<=", ">", "="]:
        if op in cond_str:
            return cond_str.split(op, 1)[0].strip()
    # fallback: first token
    return cond_str.split()[0].strip()

generic_symptoms_cols = [
    'fever_severity', 'cough_severity', 'chest_pain_severity',
    'abdominal_pain_severity', 'fatigue_level', 'nausea'
]

LUNG_TESTS = ['pulmonary_function', 'chest_xray_score', 'sputum_neutrophil_percent', 'wbc_count']
STOMACH_TESTS =['endoscopy_score', 'gastric_ph', 'h_pylori_level', 'hemoglobin']
DIAGNOSTIC_TEST_COLS = LUNG_TESTS + STOMACH_TESTS


# ---------------------------------------------------------
# FEATURE DOMAINS (for procedural vs diagnostic anchors)
# ---------------------------------------------------------
# Build a domain of possible values for each feature used by anchors.
# We take distinct non-NaN values from X_train (binned space).
FEATURE_DOMAINS = {}
for col in X_train.columns:
    vals = sorted(X_train[col].dropna().unique())
    if len(vals) > 0:
        FEATURE_DOMAINS[col] = vals

PROCEDURAL_THRESHOLD = 0.8  # you can tune this

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
# Helpers: predicate coverage in feature domain
# ---------------------------------------------------------
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

    # If the domain appears numeric, we handle numeric comparisons
    if is_numeric:
        domain_vals = np.array(domain, dtype=float)

        if op == "leq":
            covered = np.sum(domain_vals <= float(t_val)) #+ eps)
        elif op == "gt":
            covered = np.sum(domain_vals > float(t_val)) # + eps)
        elif op == "eq":
            covered = np.sum(np.abs(domain_vals - float(t_val)) < eps)
        else:
            return None
    else:
        # Non-numeric equality only
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

# def compute_resampled_precision(
#     predicate_names,
#     neighbors_raw_df,
#     pred_sets_names,
#     label_to_exclude,
#     n_samples=N_SAMPLES_PROCEDURAL
# ):
#     """
#     Compute a resampled precision for an anchor:

#     - Sample 'n_samples' neighbors with replacement from neighbors_raw_df.
#     - Keep only those sampled neighbors where ALL features appearing in
#       predicate_names are non-NaN.
#     - Among those, compute:
#         new_precision = (# of those with prediction set excluding label_to_exclude)
#                         / ( # of those )

#     Returns (new_precision, n_effective)
#       - new_precision is None if no effective sampled points.
#       - n_effective = number of sampled neighbors used in the fraction.
#     """
#     n_neighbors = len(neighbors_raw_df)
#     if n_neighbors == 0:
#         return None, 0

#     # Features involved in the anchor
#     feature_names = sorted({extract_feature_name(cond) for cond in predicate_names})

#     # If any feature is not found in the DataFrame, we can't compute this
#     for feat in feature_names:
#         if feat not in neighbors_raw_df.columns:
#             return None, 0

#     # Sample indices with replacement
#     sampled_idx = np.random.choice(n_neighbors, size=n_samples, replace=True)

#     # Precompute which neighbors have all those features non-NaN
#     valid_all = neighbors_raw_df[feature_names].notna().all(axis=1).to_numpy()  # shape (n_neighbors,)

#     # Restrict to sampled indices that are "valid"
#     valid_sample_mask = valid_all[sampled_idx]
#     if not np.any(valid_sample_mask):
#         return None, 0

#     effective_idx = sampled_idx[valid_sample_mask]

#     # For those effective neighbors, check if prediction set excludes the label
#     # pred_sets_names[i] is the list of labels in the prediction set for neighbor i
#     excludes_flags = [
#         (label_to_exclude not in pred_sets_names[i])
#         for i in effective_idx
#     ]

#     new_precision = float(np.mean(excludes_flags))
#     return new_precision, len(effective_idx)

def compute_resampled_precision(
    predicate_names,
    neighbors_raw_df,
    pred_sets_names,
    label_to_exclude,
    n_samples=N_SAMPLES_PROCEDURAL
):
    """
    Deterministic version (no resampling):

    - Use ALL neighbors in neighbors_raw_df.
    - Keep only those neighbors where ALL features appearing in
      predicate_names are non-NaN.
    - Among those, compute:
        new_precision = (# of those with prediction set excluding label_to_exclude)
                        / ( # of those )

    Returns (new_precision, n_effective)
      - new_precision is None if no effective points.
      - n_effective = number of neighbors used in the fraction.
    """
    n_neighbors = len(neighbors_raw_df)
    if n_neighbors == 0:
        return None, 0

    # Features involved in the anchor
    feature_names = sorted({extract_feature_name(cond) for cond in predicate_names})

    # If any feature is not found in the DataFrame, we can't compute this
    for feat in feature_names:
        if feat not in neighbors_raw_df.columns:
            return None, 0

    # Mask of neighbors where all these features are defined
    valid_mask = neighbors_raw_df[feature_names].notna().all(axis=1).to_numpy()  # shape (n_neighbors,)
    effective_idx = np.where(valid_mask)[0]

    if len(effective_idx) == 0:
        return None, 0

    # For those effective neighbors, check if prediction set excludes the label
    # pred_sets_names[i] is the list of labels in the prediction set for neighbor i
    excludes_flags = [
        (label_to_exclude not in pred_sets_names[i])
        for i in effective_idx
    ]

    new_precision = float(np.mean(excludes_flags))
    return new_precision, len(effective_idx)

def compute_procedural_stats_for_anchor(
    predicate_names,
    orig_precision,
    neighbors_raw_df,
    pred_sets_names,
    label_to_exclude,
    n_samples=N_SAMPLES_PROCEDURAL
):
    """
    Compute:
      - resampled precision (using only 'defined' neighbors)
      - delta = |orig_precision - resampled_precision|
      - is_procedural = (delta <= DELTA_PROCEDURAL_THRESHOLD)
    """
    new_prec, n_eff = compute_resampled_precision(
        predicate_names,
        neighbors_raw_df,
        pred_sets_names,
        label_to_exclude,
        n_samples=n_samples,
    )

    if new_prec is None or n_eff == 0:
        return None, None, False

    delta = abs(orig_precision - new_prec)
    is_procedural = (delta <= DELTA_PROCEDURAL_THRESHOLD)
    return new_prec, delta, is_procedural

# ---------------------------------------------------------
# MODEL & DATA
# ---------------------------------------------------------
xgb_cl = xgb.XGBClassifier()
xgb_cl.load_model("xgb_model.json")
class_names = xgb_cl.classes_

start_time = time.time()

# ---------------------------------------------------------
# EXPERIMENT SETTINGS
# ---------------------------------------------------------
n_instances = 2  # ONLY FIRST TWO INSTANCES
n_instances = min(n_instances, len(X_test))
beam_size = 10

# Use the first n_instances test points (no randomness)
selected_test_indices = np.random.choice(len(X_test), size=n_instances, replace=False)


# To store per-instance predicate-type counts per mode
# predicate_counts_per_instance[instance_idx][mode] = {'generic':..., 'lung':..., 'stomach':..., 'other':...}
modes_list = ['original', 'mean', 'medoid', 'union_mode1', 'union_mode2']
predicate_counts_per_instance = {
    int(idx): {m: {'generic': 0, 'lung': 0, 'stomach': 0, 'other': 0} for m in modes_list}
    for idx in selected_test_indices
}

# To store neighbor compositions for plotting
neighbor_composition = []  # list of dicts: {'instance_idx', 'n_lung', 'n_stomach'}

output_path = f"anchor_kmedoids_mean_all_labels_no_missing_1instances_10_7_tol_sorted_dom_oldapp_newproc_{time.time()}.txt"
results = []

with open(output_path, "w") as f:
    f.write("K-MEDOIDS + MEAN-INSTANCES EXPERIMENT ON ANCHORS (NO MISSINGNESS, FIRST 2 INSTANCES)\n")
    f.write("Modes: original, mean-instances, medoid, union_mode1, union_mode2\n")
    f.write(f"Number of test instances: {n_instances}\n")
    f.write(f"Beam size: {beam_size}\n")
    f.write("For each instance, ALL labels are considered as labels-to-exclude.\n")
    f.write(f"Feature domains: {FEATURE_DOMAINS}")
    f.write("=" * 100 + "\n\n")

    labels_stomach = [1, 5, 6, 7, 8]
    labels_lung = [0, 2, 3, 4, 9]

    for run_id, new_patient_idx in enumerate(selected_test_indices, 1):
        f.write(f"### INSTANCE {run_id}/{n_instances}  (test index = {new_patient_idx})\n")
        f.write("-" * 80 + "\n")

        # -----------------------------
        # 1) BASIC INSTANCE INFO (NO MISSINGNESS)
        # -----------------------------
        new_patient = X_test.iloc[new_patient_idx].copy()
        new_patient_true_label = y_test.iloc[new_patient_idx]

        new_patient_df = new_patient.to_frame().T
        new_patient_df.index = [new_patient_idx]

        # Prediction on full instance (no NaNs injected)
        new_patient_pred = xgb_cl.predict(new_patient.values.reshape(1, -1))[0]
        probs = xgb_cl.predict_proba(new_patient.values.reshape(1, -1))[0]

        f.write(f"New patient: {new_patient}\n")
        f.write(f"True label: {new_patient_true_label} ({categories[new_patient_true_label]})\n")
        f.write(f"Predicted label: {new_patient_pred} "
                f"({categories[new_patient_pred]})\n")

        f.write("Model class probabilities for test instance:\n")
        for cls_idx, p in enumerate(probs):
            f.write(
                f"  P(y={cls_idx} | x) = {p:.5f} "
                f"({categories[int(cls_idx)]})\n"
            )

        # Sum probabilities over lung-related vs stomach-related diseases
        prob_lung = float(sum(probs[l] for l in labels_lung))
        prob_stomach = float(sum(probs[l] for l in labels_stomach))

        f.write(f"Sum of probabilities over LUNG diseases: {prob_lung:.5f}\n")
        f.write(f"Sum of probabilities over STOMACH diseases: {prob_stomach:.5f}\n\n")

        # -----------------------------
        # NEIGHBORHOOD (k-NN) FOR THIS INSTANCE
        # -----------------------------
        subsets_list, y_subsets_list, similarities_list, indices_neighbors_list = get_knn_subsets(
            new_patient_df, X_anchors, y_anchors, k=100
        )

        subset_new_patient = subsets_list[0]             # DataFrame of neighbors (binned)
        y_subset_new_patient = y_subsets_list[0]         # Series of neighbor labels
        similarity_list = similarities_list[0]           # similarities for this patient
        indices_neighbors = indices_neighbors_list[0]    # indices in X_anchors_orig / y_anchors_orig

        subset_new_patient_raw = X_anchors_orig.iloc[indices_neighbors].reset_index(drop=True)
        y_subset_new_patient_raw = y_anchors_orig.iloc[indices_neighbors].reset_index(drop=True)

        # Neighbor label composition (lung vs stomach)
        n_neighbors = len(y_subset_new_patient)
        n_lung_neighbors = int(np.isin(y_subset_new_patient, labels_lung).sum())
        n_stomach_neighbors = int(np.isin(y_subset_new_patient, labels_stomach).sum())

        f.write("Neighbor label composition (k-NN subset):\n")
        f.write(f"  Total neighbors: {n_neighbors}\n")
        f.write(f"  Lung-disease neighbors:    {n_lung_neighbors}\n")
        f.write(f"  Stomach-disease neighbors: {n_stomach_neighbors}\n\n")

        neighbor_composition.append({
            'instance_idx': int(new_patient_idx),
            'n_lung': n_lung_neighbors,
            'n_stomach': n_stomach_neighbors
        })

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
        # 3) LABELS TO EXCLUDE: ALL LABELS
        # -----------------------------
        
        labels_to_exclude = [int(l) for l in class_names]
        f.write(f"Labels considered for exclusion (ALL labels): "
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

            # Here we do NOT overwrite generic symptoms (no missingness)
            anchor_instance = anchor_row.to_numpy()

            f.write(f"Anchor candidate index within subset: {idx_example_to_anchor}\n")
            f.write(f"True label of anchor neighbor: "
                    f"{y_subset_new_patient.iloc[idx_example_to_anchor]}\n")
            f.write(f"Predicted label of anchor instance: "
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
            # Overwrite generic symptoms in mean instances with patient's symptoms (no NaN)
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
            # Overwrite generic symptoms with patient's symptoms
            final_medoids_binned_df[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
            final_medoids_binned = final_medoids_binned_df.to_numpy()

            # -----------------------------
            # 7) BUILD EXPLAINERS
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
            # 8) ORIGINAL MODE
            # ----------------------------------------------------
            f.write("ORIGINAL MODE\n")

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

            main_pred_counts = count_predicate_types(exp_original.names())
            f.write(
                "Main anchor predicate composition (original): "
                f"generic={main_pred_counts['generic']}, "
                f"lung_tests={main_pred_counts['lung']}, "
                f"stomach_tests={main_pred_counts['stomach']}, "
                f"other={main_pred_counts['other']}\n"
            )

            main_names_orig = exp_original.names()
            main_applies_orig = anchor_applies_to_instance(main_names_orig, new_patient)#, use_healthy_for_nan=True)
            proc_score_main_orig = anchor_procedural_score(main_names_orig, FEATURE_DOMAINS)
            proc_prec_main_orig, delta_main_orig, is_procedural_main_orig = (
                compute_procedural_stats_for_anchor(
                    main_names_orig,
                    orig_precision=exp_original.precision(),
                    neighbors_raw_df=subset_new_patient_raw,   # ALL 100 neighbors, raw space
                    pred_sets_names=pred_sets_names,
                    label_to_exclude=label_to_exclude,
                )
            )


            f.write(f"Does MAIN anchor (original) apply to patient? "
                    f"{main_applies_orig}\n")
            
            
            f.write(
                "Procedural score (main, original): "
                f"{'%.3f' % proc_score_main_orig if proc_score_main_orig is not None else 'NA'}; "
            )
            f.write(
                "Procedural precision (resampled, main, original): "
                f"{'%.3f' % proc_prec_main_orig if proc_prec_main_orig is not None else 'NA'}, "
                f"delta= {'%.3f' % delta_main_orig if delta_main_orig is not None else 'NA'}; "
                f"is_procedural={is_procedural_main_orig}\n"
            )

            num_valid_apply_orig = 0
            for va in valid_anchors_original:
                if anchor_applies_to_instance(va['names'], new_patient): #, use_healthy_for_nan=True):
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
                applies = anchor_applies_to_instance(va['names'], new_patient)#, use_healthy_for_nan=True)
                pred_counts = count_predicate_types(va['names'])
                proc_score = anchor_procedural_score(va['names'], FEATURE_DOMAINS)
                proc_prec, delta_proc, is_procedural = compute_procedural_stats_for_anchor(
                    va['names'],
                    orig_precision=prec,  # prec = va['precision'][-1] above
                    neighbors_raw_df=subset_new_patient_raw,
                    pred_sets_names=pred_sets_names,
                    label_to_exclude=label_to_exclude,
                )

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
                f.write(
                    f"     procedural_score={ '%.3f' % proc_score if proc_score is not None else 'NA' }, "
                )
                f.write(
                f"     procedural_resampled_precision={ '%.3f' % proc_prec if proc_prec is not None else 'NA' }, "
                f"delta={ '%.3f' % delta_proc if delta_proc is not None else 'NA' }, "
                f"is_procedural={is_procedural}\n"
            )

            f.write("\n")

            # --- Aggregate predicate counts for ORIGINAL mode (this instance) ---
            all_anchor_names_orig = [exp_original.names()] + [va['names'] for va in valid_anchors_original]
            for names in all_anchor_names_orig:
                c = count_predicate_types(names)
                for k in ['generic', 'lung', 'stomach', 'other']:
                    predicate_counts_per_instance[int(new_patient_idx)]['original'][k] += c[k]

            results.append({
                'instance_idx': int(new_patient_idx),
                'label_to_exclude': int(label_to_exclude),
                'mode': 'original',
                'main_precision': exp_original.precision(),
                'main_coverage': exp_original.coverage(),
                'cumulative_coverage': exp_original.cumulative_coverage(),
                'avg_feats_valid': avg_feats,
                'num_unique_feats_valid': num_unique_feats,
                'num_valid_anchors': len(valid_anchors_original),
                'main_applies': int(main_applies_orig),
                'num_valid_apply': num_valid_apply_orig,
                'main_procedural_score': proc_score_main_orig,
                'procedural_resampled_precision_main': proc_prec_main_orig,
                'procedural_delta_main': delta_main_orig,
                'main_is_procedural': int(is_procedural_main_orig),

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

            main_pred_counts = count_predicate_types(exp_mean.names())
            f.write(
                "Main anchor predicate composition (mean-instances): "
                f"generic={main_pred_counts['generic']}, "
                f"lung_tests={main_pred_counts['lung']}, "
                f"stomach_tests={main_pred_counts['stomach']}, "
                f"other={main_pred_counts['other']}\n"
            )

            main_names_mean = exp_mean.names()
            main_applies_mean = anchor_applies_to_instance(main_names_mean, new_patient)#, use_healthy_for_nan=True)
            proc_score_main_mean = anchor_procedural_score(main_names_mean, FEATURE_DOMAINS)
            proc_prec_main_mean, delta_main_mean, is_procedural_main_mean = (
                compute_procedural_stats_for_anchor(
                    main_names_mean,
                    orig_precision=exp_mean.precision(),
                    neighbors_raw_df=subset_new_patient_raw,
                    pred_sets_names=pred_sets_names,
                    label_to_exclude=label_to_exclude,
                )
            )
            f.write(f"Does MAIN anchor (mean-instances) apply to patient? "
                    f"{main_applies_mean}\n")
            f.write(
                "Procedural score (main, mean-instances): "
                f"{'%.3f' % proc_score_main_mean if proc_score_main_mean is not None else 'NA'}; "
            )
            f.write(
                "Procedural precision (resampled, main, mean-instances): "
                f"{'%.3f' % proc_prec_main_mean if proc_prec_main_mean is not None else 'NA'}, "
                f"delta={ '%.3f' % delta_main_mean if delta_main_mean is not None else 'NA' }; "
                f"is_procedural={is_procedural_main_mean}\n"
            )

            num_valid_apply_mean = 0
            for va in valid_anchors_mean:
                if anchor_applies_to_instance(va['names'], new_patient): #, use_healthy_for_nan=True):
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
                applies = anchor_applies_to_instance(va['names'], new_patient)#, use_healthy_for_nan=True)
                pred_counts = count_predicate_types(va['names'])
                proc_score = anchor_procedural_score(va['names'], FEATURE_DOMAINS)
                proc_prec, delta_proc, is_procedural = compute_procedural_stats_for_anchor(
                    va['names'],
                    orig_precision=prec,  # prec = va['precision'][-1] above
                    neighbors_raw_df=subset_new_patient_raw,
                    pred_sets_names=pred_sets_names,
                    label_to_exclude=label_to_exclude,
                )
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
                f.write(
                    f"     procedural_score={ '%.3f' % proc_score if proc_score is not None else 'NA' }, "
                )
                f.write(
                    f"     procedural_resampled_precision={ '%.3f' % proc_prec if proc_prec is not None else 'NA' }, "
                    f"delta={ '%.3f' % delta_proc if delta_proc is not None else 'NA' }, "
                    f"is_procedural={is_procedural}\n"
                )

            f.write("\n")

            # --- Aggregate predicate counts for MEAN mode ---
            all_anchor_names_mean = [exp_mean.names()] + [va['names'] for va in valid_anchors_mean]
            for names in all_anchor_names_mean:
                c = count_predicate_types(names)
                for k in ['generic', 'lung', 'stomach', 'other']:
                    predicate_counts_per_instance[int(new_patient_idx)]['mean'][k] += c[k]

            results.append({
                'instance_idx': int(new_patient_idx),
                'label_to_exclude': int(label_to_exclude),
                'mode': 'mean',
                'main_precision': exp_mean.precision(),
                'main_coverage': exp_mean.coverage(),
                'cumulative_coverage': exp_mean.cumulative_coverage(),
                'avg_feats_valid': avg_feats_m,
                'num_unique_feats_valid': num_unique_feats_m,
                'num_valid_anchors': len(valid_anchors_mean),
                'main_applies': int(main_applies_mean),
                'num_valid_apply': num_valid_apply_mean,
                'main_procedural_score': proc_score_main_mean,
                'procedural_resampled_precision_main': proc_prec_main_mean,
                'procedural_delta_main': delta_main_mean,
                'main_is_procedural': int(is_procedural_main_mean),

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
                    beam_size= 7,#beam_size,
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

                proc_score_main_med = anchor_procedural_score(exp_m.names(), FEATURE_DOMAINS)
                proc_prec_main_med, delta_main_med, is_procedural_main_med = (
                    compute_procedural_stats_for_anchor(
                        exp_m.names(),
                        orig_precision=exp_m.precision(),
                        neighbors_raw_df=subset_new_patient_raw,
                        pred_sets_names=pred_sets_names,
                        label_to_exclude=label_to_exclude,
                    )
                )
                f.write(
                    "Procedural score (main, medoid %d): %s\n"
                    % (
                        m_id,
                        '%.3f' % proc_score_main_med if proc_score_main_med is not None else 'NA'
                    )
                )
                f.write(
                    "Procedural precision (resampled, main, medoid %d): %s, delta=%s; is_procedural=%s\n"
                    % (
                        m_id,
                        '%.3f' % proc_prec_main_med if proc_prec_main_med is not None else 'NA',
                        '%.3f' % delta_main_med if delta_main_med is not None else 'NA',
                        str(is_procedural_main_med)
                    )
                )

                main_pred_counts = count_predicate_types(exp_m.names())
                f.write(
                    "Main anchor predicate composition (medoid): "
                    f"generic={main_pred_counts['generic']}, "
                    f"lung_tests={main_pred_counts['lung']}, "
                    f"stomach_tests={main_pred_counts['stomach']}, "
                    f"other={main_pred_counts['other']}\n"
                )

                main_applies = anchor_applies_to_instance(exp_m.names(), new_patient)#, use_healthy_for_nan=True)
                f.write(f"Does MAIN anchor (medoid {m_id}) apply to patient? {main_applies}\n")

                num_valid_apply = 0
                for va in valid_anchors_m:
                    if anchor_applies_to_instance(va['names'], new_patient): #, use_healthy_for_nan=True):
                        num_valid_apply += 1
                f.write(
                    f"#valid anchors (medoid {m_id}) applying to patient: "
                    f"{num_valid_apply} / {len(valid_anchors_m)}\n"
                )

                if len(valid_anchors_m) > 0:
                    num_feats_each = [len(va['feature']) for va in valid_anchors_m]
                    avg_feats_m_med = float(np.mean(num_feats_each))
                    unique_feats_m_med = set()
                    for va in valid_anchors_m:
                        unique_feats_m_med |= set(va['feature'])
                    num_unique_feats_m_med = len(unique_feats_m_med)
                else:
                    avg_feats_m_med = 0.0
                    num_unique_feats_m_med = 0

                f.write(f"Valid anchors (medoid {m_id}): {len(valid_anchors_m)}\n")
                f.write(f"Avg #features per valid anchor (medoid {m_id}): {avg_feats_m_med:.3f}\n")
                f.write(f"#unique features across valid anchors (medoid {m_id}): "
                        f"{num_unique_feats_m_med}\n")

                sorted_valid = sorted(
                    valid_anchors_m,
                    key=lambda va: va['coverage'][-1],
                    reverse=True
                )
                for j, va in enumerate(sorted_valid, 1):
                    names = " AND ".join(va['names'])
                    prec = va['precision'][-1]
                    cov = va['coverage'][-1]
                    applies = anchor_applies_to_instance(va['names'], new_patient)#, use_healthy_for_nan=True)
                    pred_counts = count_predicate_types(va['names'])
                    proc_score = anchor_procedural_score(va['names'], FEATURE_DOMAINS)
                    proc_prec, delta_proc, is_procedural = compute_procedural_stats_for_anchor(
                        va['names'],
                        orig_precision=prec,  # prec = va['precision'][-1]
                        neighbors_raw_df=subset_new_patient_raw,
                        pred_sets_names=pred_sets_names,
                        label_to_exclude=label_to_exclude,
                    )


                    f.write(
                        f"  {j}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}, "
                        f"applies_to_patient={applies}\n"
                    )
                    f.write(
                        f"     predicate breakdown: "
                        f"generic={pred_counts['generic']}, "
                        f"lung_tests={pred_counts['lung']}, "
                        f"stomach_tests={pred_counts['stomach']}, "
                        f"other={pred_counts['other']}\n"
                    )
                    f.write(
                        f"     procedural_score={ '%.3f' % proc_score if proc_score is not None else 'NA' }, "
                    )
                    f.write(
                        f"     procedural_resampled_precision={ '%.3f' % proc_prec if proc_prec is not None else 'NA' }, "
                        f"delta={ '%.3f' % delta_proc if delta_proc is not None else 'NA' }, "
                        f"is_procedural={is_procedural}\n"
                    )

                f.write("\n")

                results.append({
                    'instance_idx': int(new_patient_idx),
                    'label_to_exclude': int(label_to_exclude),
                    'mode': 'medoid',
                    'medoid_id': m_id,
                    'main_precision': exp_m.precision(),
                    'main_coverage': exp_m.coverage(),
                    'cumulative_coverage': exp_m.cumulative_coverage(),
                    'avg_feats_valid': avg_feats_m_med,
                    'num_unique_feats_valid': num_unique_feats_m_med,
                    'num_valid_anchors': len(valid_anchors_m),
                    'main_applies': int(main_applies),
                    'num_valid_apply': num_valid_apply,
                })

            # --- Aggregate predicate counts for MEDOID mode (all medoid anchors for this instance+label) ---
            for a in all_medoid_anchors_for_instance:
                c = count_predicate_types(a['names'])
                for k in ['generic', 'lung', 'stomach', 'other']:
                    predicate_counts_per_instance[int(new_patient_idx)]['medoid'][k] += c[k]

            # ----------------------------------------------------
            # 11) UNION-PRUNED FINAL ANCHORS – MODE 1
            # ----------------------------------------------------
            f.write("\nUNION-PRUNED FINAL ANCHORS (across all medoids) – "
                    "MODE 1 (max marginal gain):\n")

            coverage_df = subset_new_patient
            flatten_tol = 1e-2

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

            # def extract_feature_name(cond_str):
            #     cond_str = cond_str.strip()
            #     for op in ["≤", "<=", ">", "="]:
            #         if op in cond_str:
            #             return cond_str.split(op, 1)[0].strip()
            #     return cond_str

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
                    pred_counts = count_predicate_types(fa['names'])
                    f.write(
                        f"     predicate breakdown (union MODE 1): "
                        f"generic={pred_counts['generic']}, "
                        f"lung_tests={pred_counts['lung']}, "
                        f"stomach_tests={pred_counts['stomach']}, "
                        f"other={pred_counts['other']}\n"
                    )
                    proc_score = anchor_procedural_score(fa['names'], FEATURE_DOMAINS)
                    proc_prec, delta_proc, is_procedural = compute_procedural_stats_for_anchor(
                        fa['names'],
                        orig_precision=fa['precision'],
                        neighbors_raw_df=subset_new_patient_raw,
                        pred_sets_names=pred_sets_names,
                        label_to_exclude=label_to_exclude,
                    )
                    fa['procedural_score'] = proc_score
                    fa['procedural_resampled_precision'] = proc_prec
                    fa['procedural_delta'] = delta_proc
                    fa['is_procedural'] = is_procedural

                    f.write(
                        f"     procedural_score={ '%.3f' % proc_score if proc_score is not None else 'NA' }, "
                    )
                    f.write(
                        f"     procedural_resampled_precision={ '%.3f' % proc_prec if proc_prec is not None else 'NA' }, "
                        f"delta={ '%.3f' % delta_proc if delta_proc is not None else 'NA' }, "
                        f"is_procedural={is_procedural}\n"
                    )
                f.write("\n")

                # Aggregate predicate counts for UNION MODE 1
                for fa in final_anchors_union1:
                    c = count_predicate_types(fa['names'])
                    for k in ['generic', 'lung', 'stomach', 'other']:
                        predicate_counts_per_instance[int(new_patient_idx)]['union_mode1'][k] += c[k]

                main_union_anchor1 = final_anchors_union1[0]
                main_prec_union1 = main_union_anchor1['precision']
                main_cov_union1 = main_union_anchor1['coverage']
                main_proc_score_union1 = main_union_anchor1.get('procedural_score', None)
                main_proc_prec_union1 = main_union_anchor1.get('procedural_resampled_precision', None)
                main_delta_union1 = main_union_anchor1.get('procedural_delta', None)
                main_is_proc_union1 = main_union_anchor1.get('is_procedural', False)
   
                f.write(f"  Main union-anchor procedural score: "
                        f"{'%.3f' % main_proc_score_union1 if main_proc_score_union1 is not None else 'NA'}\n")
                f.write(f"  Main union-anchor procedural resampled precision: "
                        f"{'%.3f' % main_proc_prec_union1 if main_proc_prec_union1 is not None else 'NA'}\n")
                f.write(f"  Main union-anchor procedural delta: "
                        f"{'%.3f' % main_delta_union1 if main_delta_union1 is not None else 'NA'}\n")
                f.write(f"  Main union-anchor is_procedural: {main_is_proc_union1}\n")

                num_feats_each_union1 = [len(a['names']) for a in final_anchors_union1]
                avg_feats_union1 = float(np.mean(num_feats_each_union1))

                unique_feature_names_union1 = set()
                for a in final_anchors_union1:
                    for cond in a['names']:
                        unique_feature_names_union1.add(extract_feature_name(cond))
                num_unique_feats_union1 = len(unique_feature_names_union1)

                main_applies_union1 = int(
                    anchor_applies_to_instance(main_union_anchor1['names'], new_patient)#, use_healthy_for_nan=True)
                )
                num_valid_apply_union1 = 0
                for a in final_anchors_union1:
                    if anchor_applies_to_instance(a['names'], new_patient): #, use_healthy_for_nan=True):
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
                    'instance_idx': int(new_patient_idx),
                    'label_to_exclude': int(label_to_exclude),
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
                    pred_counts = count_predicate_types(fa['names'])
                    f.write(
                        f"     predicate breakdown (union MODE 2): "
                        f"generic={pred_counts['generic']}, "
                        f"lung_tests={pred_counts['lung']}, "
                        f"stomach_tests={pred_counts['stomach']}, "
                        f"other={pred_counts['other']}\n"
                    )
                    proc_score = anchor_procedural_score(fa['names'], FEATURE_DOMAINS)
                    proc_prec, delta_proc, is_procedural = compute_procedural_stats_for_anchor(
                        fa['names'],
                        orig_precision=fa['precision'],
                        neighbors_raw_df=subset_new_patient_raw,
                        pred_sets_names=pred_sets_names,
                        label_to_exclude=label_to_exclude,
                    )

                    fa['procedural_score'] = proc_score
                    fa['procedural_resampled_precision'] = proc_prec
                    fa['procedural_delta'] = delta_proc
                    fa['is_procedural'] = is_procedural

                    f.write(
                        f"     procedural_score={ '%.3f' % proc_score if proc_score is not None else 'NA' }, "
                    )
                    f.write(
                        f"     procedural_resampled_precision={ '%.3f' % proc_prec if proc_prec is not None else 'NA' }, "
                        f"delta={ '%.3f' % delta_proc if delta_proc is not None else 'NA' }, "
                        f"is_procedural={is_procedural}\n"
                    )
                f.write("\n")

                # Aggregate predicate counts for UNION MODE 2
                for fa in final_anchors_union2:
                    c = count_predicate_types(fa['names'])
                    for k in ['generic', 'lung', 'stomach', 'other']:
                        predicate_counts_per_instance[int(new_patient_idx)]['union_mode2'][k] += c[k]

                main_union_anchor2 = final_anchors_union2[0]
                main_prec_union2 = main_union_anchor2['precision']
                main_cov_union2 = main_union_anchor2['coverage']
                main_proc_score_union2 = main_union_anchor2.get('procedural_score', None)
                main_proc_prec_union2 = main_union_anchor2.get('procedural_resampled_precision', None)
                main_delta_union2 = main_union_anchor2.get('procedural_delta', None)
                main_is_proc_union2 = main_union_anchor2.get('is_procedural', False)

                f.write(f"  Main union-anchor procedural score: "
                        f"{'%.3f' % main_proc_score_union2 if main_proc_score_union2 is not None else 'NA'}\n")
                f.write(f"  Main union-anchor procedural resampled precision: "
                        f"{'%.3f' % main_proc_prec_union2 if main_proc_prec_union2 is not None else 'NA'}\n")
                f.write(f"  Main union-anchor procedural delta: "
                        f"{'%.3f' % main_delta_union2 if main_delta_union2 is not None else 'NA'}\n")
                f.write(f"  Main union-anchor is_procedural: {main_is_proc_union2}\n")

                num_feats_each_union2 = [len(a['names']) for a in final_anchors_union2]
                avg_feats_union2 = float(np.mean(num_feats_each_union2))

                unique_feature_names_union2 = set()
                for a in final_anchors_union2:
                    for cond in a['names']:
                        unique_feature_names_union2.add(extract_feature_name(cond))
                num_unique_feats_union2 = len(unique_feature_names_union2)

                main_applies_union2 = int(
                    anchor_applies_to_instance(main_union_anchor2['names'], new_patient)#, use_healthy_for_nan=True)
                )
                num_valid_apply_union2 = 0
                for a in final_anchors_union2:
                    if anchor_applies_to_instance(a['names'], new_patient): #, use_healthy_for_nan=True):
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
                    'instance_idx': int(new_patient_idx),
                    'label_to_exclude': int(label_to_exclude),
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
    # 13) GLOBAL STATISTICS (you can keep as in your script, or trim)
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
# 14) PLOTS: NEIGHBORHOOD COMPOSITION + PREDICATE DISTRIBUTIONS
# ---------------------------------------------------------
# Neighborhood composition plots (lung vs stomach) per instance
for comp in neighbor_composition:
    inst_idx = comp['instance_idx']
    n_lung = comp['n_lung']
    n_stomach = comp['n_stomach']

    plt.figure(figsize=(3, 3))
    plt.bar(['Lung', 'Stomach'], [n_lung, n_stomach], color=['tab:blue', 'orange'])
    plt.ylabel('Number of neighbors')
    plt.title(f'Neighborhood composition')
    plt.tight_layout()
    plt.savefig(f'neighborhood_composition_instance_{inst_idx}.pdf')
    plt.close()

# Predicate-type distribution per instance & mode (stacked bar)
for inst_idx in selected_test_indices:
    inst_idx = int(inst_idx)
    counts_by_mode = predicate_counts_per_instance[inst_idx]

    #modes_plot = modes_list
    modes_plot = ['original', 'mean', 'union_mode1', 'union_mode2']

    x = np.arange(len(modes_plot))

    generic_counts = [counts_by_mode[m]['generic'] for m in modes_plot]
    lung_counts = [counts_by_mode[m]['lung'] for m in modes_plot]
    stomach_counts = [counts_by_mode[m]['stomach'] for m in modes_plot]
    #other_counts = [counts_by_mode[m]['other'] for m in modes_plot]

    plt.figure(figsize=(6, 4))
    plt.bar(x, generic_counts, label='Generic', color='lightgray')
    plt.bar(x, lung_counts, bottom=generic_counts, label='Lung tests', color='tab:blue')
    bottom_g_l = np.array(generic_counts) + np.array(lung_counts)
    plt.bar(x, stomach_counts, bottom=bottom_g_l, label='Stomach tests', color='orange')
    bottom_all = bottom_g_l + np.array(stomach_counts)
    #plt.bar(x, other_counts, bottom=bottom_all, label='Other')

    plt.xticks(x, modes_plot, rotation=30)
    plt.ylabel('Number of predicates')
    plt.title(f'Predicate-type distribution by modality')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'predicate_type_distribution_instance_{inst_idx}.pdf')
    plt.close()

end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"\nDone. Full report saved to: {output_path}")
