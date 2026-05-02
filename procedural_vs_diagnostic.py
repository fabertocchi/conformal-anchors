# ---------------------------------------------------------
# EXPERIMENT: PROCEDURAL vs DIAGNOSTIC ANCHORS (ALL MODES)
# - Sample up to 100 test instances
# - For each instance:
#     * pick one within-group label to exclude
#     * pick one cross-group label to exclude
#     * build anchors with ALL modalities:
#         - original
#         - mean-instances
#         - medoid
#         - union pruning mode 1
#         - union pruning mode 2
#     * compute procedural score for each anchor
# - Plot distributions of procedural scores (all anchors)
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
# BASIC DEFINITIONS
# ---------------------------------------------------------
generic_symptoms_cols = [
    'fever_severity', 'cough_severity', 'chest_pain_severity',
    'abdominal_pain_severity', 'fatigue_level', 'nausea'
]

LUNG_TESTS = ['pulmonary_function', 'chest_xray_score', 'sputum_neutrophil_percent', 'wbc_count']
STOMACH_TESTS = ['endoscopy_score', 'gastric_ph', 'h_pylori_level', 'hemoglobin']
DIAGNOSTIC_TEST_COLS = LUNG_TESTS + STOMACH_TESTS

# Lung vs stomach labels (as in your project)
labels_stomach = [1, 5, 6, 7, 8]
labels_lung = [0, 2, 3, 4, 9]

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
xgb_cl = xgb.XGBClassifier()
xgb_cl.load_model("xgb_model.json")
class_names = xgb_cl.classes_

# ---------------------------------------------------------
# FEATURE DOMAINS (for procedural score)
# ---------------------------------------------------------
FEATURE_DOMAINS = {}
for col in X_train.columns:
    vals = sorted(X_train[col].dropna().unique())
    if len(vals) > 0:
        FEATURE_DOMAINS[col] = vals

PROCEDURAL_THRESHOLD = 0.8  # only used for reference in plots

# ---------------------------------------------------------
# GOWER DISTANCE HELPERS (RAW)
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
# PREDICATE PARSING & PROCEDURAL SCORE
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

    if is_numeric:
        domain_vals = np.array(domain, dtype=float)

        if op == "leq":
            covered = np.sum(domain_vals <= float(t_val) + eps)
        elif op == "gt":
            covered = np.sum(domain_vals > float(t_val) + eps)
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
# UNION PRUNING – MODE 1 (max marginal gain)
# ---------------------------------------------------------
def union_prune_anchors_mode1(anchors, coverage_df, n_samples=10000, flatten_tol=1e-6):
    """
    anchors: list of dicts with at least 'names' (list of predicates)
    coverage_df: DataFrame in SAME feature space as anchors (binned)
    """
    if not anchors:
        return [], 0.0, [], []

    n_universe = coverage_df.shape[0]
    if n_universe == 0:
        return [], 0.0, [], []

    sampled_idx = np.random.choice(range(n_universe), size=n_samples, replace=True)
    coverage_data = coverage_df.iloc[sampled_idx].reset_index(drop=True)
    n_cov = coverage_data.shape[0]

    # precompute which sampled rows each anchor covers
    state = {'t_coverage_idx': {}}
    for t, a in enumerate(anchors):
        mask = coverage_data.apply(
            lambda row: _anchor_applies_to_row(a['names'], row),
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


def union_prune_anchors_mode2(anchors, coverage_df, n_samples=10000, flatten_tol=1e-6):
    """
    MODE 2: sort anchors by individual coverage, then add them greedily
    as long as they increase union coverage by more than `flatten_tol`.
    Uses the same _anchor_applies_to_row logic as mode 1.
    """
    if not anchors:
        return [], 0.0, [], []

    n_universe = coverage_df.shape[0]
    if n_universe == 0:
        return [], 0.0, [], []

    sampled_idx = np.random.choice(range(n_universe), size=n_samples, replace=True)
    coverage_data = coverage_df.iloc[sampled_idx].reset_index(drop=True)
    n_cov = coverage_data.shape[0]

    # Precompute coverage for each anchor on the sampled universe
    state = {'t_coverage_idx': {}}
    for t, a in enumerate(anchors):
        mask = coverage_data.apply(
            lambda row: _anchor_applies_to_row(a['names'], row),
            axis=1
        ).to_numpy(dtype=bool)

        covered_idx = set(np.where(mask)[0])
        state['t_coverage_idx'][t] = covered_idx
        a['coverage_idx'] = covered_idx
        a['cov_train'] = float(len(covered_idx)) / n_cov

    # Sort anchors by individual coverage (descending)
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


def _anchor_applies_to_row(predicate_names, row):
    """
    Simple check: does an anchor's predicates hold on a pandas Series row?
    Uses same parsing as above, but without missingness tricks (NaN -> False).
    """
    for cond in predicate_names:
        cond = cond.strip()
        feature, op, t_val, is_numeric = parse_condition(cond)
        if feature is None or feature not in row.index:
            return False

        x = row[feature]
        if pd.isna(x):
            return False

        if is_numeric:
            x_val = float(x)
            if op == "leq":
                if not (x_val <= float(t_val)):
                    return False
            elif op == "gt":
                if not (x_val > float(t_val)):
                    return False
            elif op == "eq":
                if not (abs(x_val - float(t_val)) < 1e-20):
                    return False
            else:
                return False
        else:
            if op != "eq":
                return False
            if str(x) != str(t_val):
                return False

    return True

# ---------------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------------
start_time = time.time()

n_instances = 200
n_instances = min(n_instances, len(X_test))
beam_size = 10
flatten_tol = 1e-2

# random sample of test indices
selected_test_indices = np.random.choice(len(X_test), size=n_instances, replace=False)

anchor_records = []  # rows: instance_idx, label_to_exclude, context, mode, procedural_score, ...

for run_id, new_patient_idx in enumerate(selected_test_indices, 1):
    new_patient_idx = int(new_patient_idx)
    new_patient = X_test.iloc[new_patient_idx].copy()
    true_label = int(y_test.iloc[new_patient_idx])

    # Determine disease group of the true label
    if true_label in labels_lung:
        same_group_labels = [l for l in labels_lung if l != true_label]
        cross_group_labels = labels_stomach
    elif true_label in labels_stomach:
        same_group_labels = [l for l in labels_stomach if l != true_label]
        cross_group_labels = labels_lung
    else:
        # if any label is outside the two groups, skip this instance
        continue

    if len(same_group_labels) == 0 or len(cross_group_labels) == 0:
        continue

    # pick one random same-group label and one cross-group label
    label_same = int(np.random.choice(same_group_labels))
    label_cross = int(np.random.choice(cross_group_labels))

    # Build a 1-row DataFrame for this patient (for k-NN function)
    new_patient_df = new_patient.to_frame().T
    new_patient_df.index = [new_patient_idx]

    # NEIGHBORHOOD via k-NN
    subsets_list, y_subsets_list, similarities_list, indices_neighbors_list = get_knn_subsets(
        new_patient_df, X_anchors, y_anchors, k=100
    )

    subset_new_patient = subsets_list[0]             # binned neighbors
    y_subset_new_patient = y_subsets_list[0]         # neighbor labels
    indices_neighbors = indices_neighbors_list[0]    # indices in X_anchors_orig / y_anchors_orig

    subset_new_patient_raw = X_anchors_orig.iloc[indices_neighbors].reset_index(drop=True)
    neighbors_labels = y_anchors_orig.iloc[indices_neighbors].reset_index(drop=True)

    # CONFORMAL PREDICTION SETS on neighborhood
    alpha = 0.01
    prediction_sets, qhat = compute_conformal_prediction_set_batch(
        xgb_cl,
        X_conf_pred,
        y_conf_pred,
        subset_new_patient,
        alpha=alpha
    )

    pred_sets_names = [list(class_names[mask]) for mask in prediction_sets]

    # ---------------- Helper: process one label (within/cross) ----------------
    def process_label(label_to_exclude, context_tag):
        # find neighbors whose prediction sets exclude this label
        pred_set_without_target = {}
        for i, pred_set in enumerate(pred_sets_names):
            if label_to_exclude not in pred_set:
                pred_set_without_target[i] = pred_set

        if len(pred_set_without_target) == 0:
            return

        idxs_neighbors_excluding_target = list(pred_set_without_target.keys())
        neighbors_excl_binned = subset_new_patient.iloc[idxs_neighbors_excluding_target]
        neighbors_excl_raw = subset_new_patient_raw.iloc[idxs_neighbors_excluding_target]
        labels_excl = neighbors_labels.iloc[idxs_neighbors_excluding_target]
        unique_labels = np.unique(labels_excl)

        # If we have too few neighbors, skip
        if len(unique_labels) == 0 or len(neighbors_excl_binned) < 2:
            return

        # -------------------------------------------------
        # 1) MEAN INSTANCES (raw, per label)
        # -------------------------------------------------
        mean_instances_raw = []
        cluster_indices_list = []
        for lab in unique_labels:
            mask_lab = (labels_excl == lab)
            cluster = neighbors_excl_raw.loc[mask_lab]
            cluster_indices = np.where(mask_lab.values)[0]  # local indices in neighbors_excl_raw
            cluster_indices_list.append(cluster_indices)
            mean_instances_raw.append(cluster.mean(axis=0))
        mean_instances_raw = np.vstack(mean_instances_raw)

        # binned mean instances & overwrite generic symptoms with patient's symptoms
        mean_instances_df = pd.DataFrame(mean_instances_raw, columns=neighbors_excl_raw.columns)
        mean_instances_binned_df = bin_dataset(
            mean_instances_df,
            TESTS=TESTS,
            generic_symptoms_cols=generic_symptoms_cols,
            verbose=False
        )
        mean_instances_binned_df[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values

        # -------------------------------------------------
        # 2) K-MEDOIDS PREPARATION (RAW Gower) FOR MEDOID MODES
        # -------------------------------------------------
        X_cluster_raw_df = neighbors_excl_raw.reset_index(drop=True)
        X_cluster_raw = X_cluster_raw_df.to_numpy(dtype=float)
        feature_names_raw = X_cluster_raw_df.columns.tolist()

        D = compute_gower_distance_matrix_raw(
            X_cluster_raw,
            feature_names=feature_names_raw,
            symptom_cols=generic_symptoms_cols,
            TEST_BOUNDS=TEST_BOUNDS,
            symptom_range=(0, 10)
        )

        # snap each mean to nearest point in its cluster -> init medoids
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
        if k == 0:
            return

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
        final_medoids_raw_df = X_cluster_raw_df.iloc[medoids_idx].copy().reset_index(drop=True)

        # Bin medoids to explainer space and overwrite generic symptoms with patient
        final_medoids_binned_df = bin_dataset(
            final_medoids_raw_df,
            TESTS=TESTS,
            generic_symptoms_cols=generic_symptoms_cols,
            verbose=False
        )
        final_medoids_binned_df[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
        final_medoids_binned = final_medoids_binned_df.to_numpy()

        # -------------------------------------------------
        # 3) BUILD EXPLAINERS
        # -------------------------------------------------
        feature_cols = X_train.columns.tolist()
        categorical_names = {}

        # ORIGINAL + MEAN-INSTANCES explainer uses neighbors_excl_binned
        if len(neighbors_excl_binned) < 2:
            return

        # choose an anchor neighbor (local index in neighbors_excl_binned)
        anchor_local_pos = np.random.choice(len(neighbors_excl_binned))
        anchor_instance_binned = neighbors_excl_binned.iloc[anchor_local_pos].to_numpy()

        # training data for original/mean modes: all other neighbors
        train_data_orig = neighbors_excl_binned.drop(neighbors_excl_binned.index[anchor_local_pos]).to_numpy()

        explainer_orig = AnchorTabularExplainer(
            class_names=class_names,
            feature_names=feature_cols,
            train_data=train_data_orig,
            discretizer=None,
            categorical_names=categorical_names,
        )

        # MEDOID explainer sees all neighbors_excl_binned
        explainer_medoid = AnchorTabularExplainer(
            class_names=class_names,
            feature_names=feature_cols,
            train_data=neighbors_excl_binned.to_numpy(),
            discretizer=None,
            categorical_names=categorical_names,
        )

        # -------------------------------------------------
        # 4) ORIGINAL MODE
        # -------------------------------------------------
        anchors_original = []

        exp_orig, valid_anchors_orig = explainer_orig.explain_instance(
            anchor_instance_binned,
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

        anchors_original.append({
            'names': exp_orig.names(),
            'precision': exp_orig.precision(),
            'coverage': exp_orig.coverage(),
        })

        for va in valid_anchors_orig:
            anchors_original.append({
                'names': va['names'],
                'precision': va['precision'][-1],
                'coverage': va['coverage'][-1],
            })

        for a in anchors_original:
            proc_score = anchor_procedural_score(a['names'], FEATURE_DOMAINS)
            if proc_score is None:
                continue
            anchor_records.append({
                'instance_idx': new_patient_idx,
                'true_label': true_label,
                'label_to_exclude': label_to_exclude,
                'context': context_tag,
                'mode': 'original',
                'procedural_score': proc_score,
                'precision': a['precision'],
                'coverage': a['coverage'],
            })

        # -------------------------------------------------
        # 5) MEAN-INSTANCES MODE
        # -------------------------------------------------
        anchors_mean = []

        exp_mean, valid_anchors_mean = explainer_orig.explain_instance(
            anchor_instance_binned,
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

        anchors_mean.append({
            'names': exp_mean.names(),
            'precision': exp_mean.precision(),
            'coverage': exp_mean.coverage(),
        })

        for va in valid_anchors_mean:
            anchors_mean.append({
                'names': va['names'],
                'precision': va['precision'][-1],
                'coverage': va['coverage'][-1],
            })

        for a in anchors_mean:
            proc_score = anchor_procedural_score(a['names'], FEATURE_DOMAINS)
            if proc_score is None:
                continue
            anchor_records.append({
                'instance_idx': new_patient_idx,
                'true_label': true_label,
                'label_to_exclude': label_to_exclude,
                'context': context_tag,
                'mode': 'mean_instances',
                'procedural_score': proc_score,
                'precision': a['precision'],
                'coverage': a['coverage'],
            })

        # -------------------------------------------------
        # 6) MEDOID MODE (per-medoid anchors, pre-union)
        # -------------------------------------------------
        all_medoid_anchors = []

        for m_id, medoid_instance in enumerate(final_medoids_binned, 1):
            exp_m, valid_anchors_m = explainer_medoid.explain_instance(
                medoid_instance,
                xgb_cl,
                mode="conformal",
                query_label=label_to_exclude,
                qhat=qhat,
                threshold=0.95,
                delta=0.1,
                tau=0.15,
                beam_size=7,
                predicate_mode="medoid",
                mean_instances=None
            )

            for va in valid_anchors_m:
                all_medoid_anchors.append({
                    'names': va['names'],
                    'medoid_id': m_id,
                    'source': 'valid_anchor',
                    'precision': va['precision'][-1],
                    'coverage': va['coverage'][-1],
                })

            main_names_m = exp_m.names()
            already_present = any(
                (a['names'] == main_names_m) and (a['medoid_id'] == m_id)
                for a in all_medoid_anchors
            )
            if not already_present:
                all_medoid_anchors.append({
                    'names': main_names_m,
                    'medoid_id': m_id,
                    'source': 'main_anchor',
                    'precision': exp_m.precision(),
                    'coverage': exp_m.coverage(),
                })

        # record MEDOID-mode anchors (before union pruning)
        for a in all_medoid_anchors:
            proc_score = anchor_procedural_score(a['names'], FEATURE_DOMAINS)
            if proc_score is None:
                continue
            anchor_records.append({
                'instance_idx': new_patient_idx,
                'true_label': true_label,
                'label_to_exclude': label_to_exclude,
                'context': context_tag,
                'mode': 'medoid',
                'procedural_score': proc_score,
                'precision': a['precision'],
                'coverage': a['coverage'],
            })

        if len(all_medoid_anchors) == 0:
            return

        coverage_df = neighbors_excl_binned

        # -------------------------------------------------
        # 7) UNION-PRUNED ANCHORS – MODE 1
        # -------------------------------------------------
        final_anchors_union1, final_union_cov1, _, _ = union_prune_anchors_mode1(
            all_medoid_anchors,
            coverage_df,
            n_samples=10000,
            flatten_tol=flatten_tol
        )

        for fa in final_anchors_union1:
            proc_score = anchor_procedural_score(fa['names'], FEATURE_DOMAINS)
            if proc_score is None:
                continue
            anchor_records.append({
                'instance_idx': new_patient_idx,
                'true_label': true_label,
                'label_to_exclude': label_to_exclude,
                'context': context_tag,
                'mode': 'union_mode1',
                'procedural_score': proc_score,
                'precision': fa['precision'],
                'coverage': fa['coverage'],
            })

        # -------------------------------------------------
        # 8) UNION-PRUNED ANCHORS – MODE 2
        # -------------------------------------------------
        final_anchors_union2, final_union_cov2, _, _ = union_prune_anchors_mode2(
            all_medoid_anchors,
            coverage_df,
            n_samples=10000,
            flatten_tol=flatten_tol
        )

        for fa in final_anchors_union2:
            proc_score = anchor_procedural_score(fa['names'], FEATURE_DOMAINS)
            if proc_score is None:
                continue
            anchor_records.append({
                'instance_idx': new_patient_idx,
                'true_label': true_label,
                'label_to_exclude': label_to_exclude,
                'context': context_tag,
                'mode': 'union_mode2',
                'procedural_score': proc_score,
                'precision': fa['precision'],
                'coverage': fa['coverage'],
            })

    # Process within-group and cross-group labels for this instance
    process_label(label_same, context_tag='within_group')
    process_label(label_cross, context_tag='cross_group')

# ---------------------------------------------------------
# BUILD DATAFRAME & PLOTS
# ---------------------------------------------------------
if len(anchor_records) == 0:
    print("No anchors collected in this experiment. Check settings.")
else:
    df_anchors = pd.DataFrame(anchor_records)
    print("Total anchors collected:", len(df_anchors))

    # Filter out any NaNs in procedural_score just in case
    df_anchors = df_anchors.dropna(subset=['procedural_score'])

    # Rounded version (choose 1 or 2 decimals depending on how coarse you want)
    df_anchors['procedural_score_rounded'] = df_anchors['procedural_score'].round(2)

    # Extract all procedural scores (ignore context/mode for now)
    scores = df_anchors['procedural_score'].values
    scores_rounded = df_anchors['procedural_score_rounded'].values

    print(f"Number of anchors with valid procedural score: {len(scores)}")
    print(f"Min score: {scores.min():.3f}, Max score: {scores.max():.3f}")
    print(f"Rounded (1 dec) min/max: {scores_rounded.min():.1f} – {scores_rounded.max():.1f}")

    # # -----------------------------------------------------
    # # (1) Bar plot: counts per unique procedural score
    # # -----------------------------------------------------
    # unique_scores, counts = np.unique(scores, return_counts=True)

    # plt.figure(figsize=(7, 4))
    # plt.bar(unique_scores, counts, width=0.02, edgecolor="black")
    # plt.xlabel('Procedural score')
    # plt.ylabel('Count')
    # plt.title('Distribution of procedural scores across all anchors\n(count per unique score)')
    # plt.xlim(0, 1)
    # plt.axvline(PROCEDURAL_THRESHOLD, linestyle='--', label=f'Threshold = {PROCEDURAL_THRESHOLD}')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('procedural_scores_bar_unique.pdf')
    # plt.close()

    # -----------------------------------------------------
    # (1) Bar plot: counts per unique procedural score
    # -----------------------------------------------------
    unique_scores, counts = np.unique(scores, return_counts=True)

    plt.figure(figsize=(7, 4))
    plt.bar(unique_scores, counts, width=0.02, edgecolor="black")
    plt.xlabel('Procedural score')
    plt.ylabel('Count')
    plt.title('Distribution of procedural scores across all anchors\n(count per unique score)')
    plt.xlim(0, 1)
    plt.axvline(PROCEDURAL_THRESHOLD, linestyle='--', label=f'Threshold = {PROCEDURAL_THRESHOLD}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('procedural_scores_bar_unique.pdf')
    plt.close()


    # # -----------------------------------------------------
    # # (2) Histogram: all scores together
    # # -----------------------------------------------------
    # bins = np.linspace(0, 1, 21)  # 20 bins on [0, 1]
    # plt.figure(figsize=(7, 4))
    # plt.hist(scores, bins=bins, alpha=0.8, edgecolor='black')
    # plt.xlabel('Procedural score')
    # plt.ylabel('Count')
    # plt.title('Histogram of procedural scores across all anchors')
    # plt.xlim(0, 1)
    # plt.axvline(PROCEDURAL_THRESHOLD, linestyle='--', label=f'Threshold = {PROCEDURAL_THRESHOLD}')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('procedural_scores_hist_all.pdf')
    # plt.close()

        # -----------------------------------------------------
    # (2) Histogram: using rounded scores
    # -----------------------------------------------------
    bins = np.arange(-0.05, 1.05 + 0.1, 0.1)  # bin per 0.1 interval

    plt.figure(figsize=(7, 4))
    plt.hist(scores_rounded, bins=bins, alpha=0.8, edgecolor='black')
    plt.xlabel('Procedural score (rounded to 0.1)')
    plt.ylabel('Count')
    plt.title('Histogram of procedural scores (rounded)')
    plt.xlim(0, 1)
    plt.axvline(PROCEDURAL_THRESHOLD, linestyle='--', label=f'Threshold = {PROCEDURAL_THRESHOLD}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('procedural_scores_hist_all_rounded.pdf')
    plt.close()

    # # -----------------------------------------------------
    # # (3) Rug / scatter plot: one point per anchor
    # # -----------------------------------------------------
    # plt.figure(figsize=(7, 1.8))
    # plt.scatter(scores, np.zeros_like(scores), alpha=0.5, s=10)
    # plt.yticks([])
    # plt.xlabel('Procedural score')
    # plt.title('Procedural score of each anchor (one point per anchor)')
    # plt.xlim(0, 1)
    # plt.axvline(PROCEDURAL_THRESHOLD, linestyle='--', label=f'Threshold = {PROCEDURAL_THRESHOLD}')
    # plt.legend(loc='upper left')
    # plt.tight_layout()
    # plt.savefig('procedural_scores_rug_all.pdf')
    # plt.close()

        # -----------------------------------------------------
    # (3) Rug / scatter plot: one point per anchor (rounded)
    # -----------------------------------------------------
    plt.figure(figsize=(7, 1.8))
    plt.scatter(scores_rounded, np.zeros_like(scores_rounded), alpha=0.5, s=10)
    plt.yticks([])
    plt.xlabel('Procedural score (rounded to 0.1)')
    plt.title('Procedural score of each anchor (one point per anchor)')
    plt.xlim(0, 1)
    plt.axvline(PROCEDURAL_THRESHOLD, linestyle='--', label=f'Threshold = {PROCEDURAL_THRESHOLD}')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('procedural_scores_rug_all_rounded.pdf')
    plt.close()

end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print("Plots saved as:")
print("  procedural_scores_bar_unique.pdf")
print("  procedural_scores_hist_all.pdf")
print("  procedural_scores_rug_all.pdf")