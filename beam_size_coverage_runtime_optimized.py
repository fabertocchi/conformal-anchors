import os
from scipy import stats

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ---------------------------------------------------------
# BEAM SIZE SENSITIVITY (UNION PRUNING MODES 1 & 2, MEDOID)
# Coverage vs beam size + Runtime vs beam size
# - 50 anchor instances
# - for each instance: exclude ONE label in the SAME disease group (≠ true), chosen at random
# - medoid-based anchors + union pruning mode 1 and 2
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
from subset import get_knn_subsets, gower_similarity_mixed_raw
from prediction_set import compute_conformal_threshold, construct_prediction_sets
from data.tests_v2 import TESTS
from binning import bin_dataset
from dataset_basic_setting import TEST_BOUNDS
from sklearn_extra.cluster import KMedoids
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import matplotlib as mpl

sns.set_theme(style="whitegrid", palette="colorblind", context="talk")
COLORBLIND_PALETTE = sns.color_palette("colorblind")

mpl.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
})

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
BASE_SEED = 1
np.random.seed(BASE_SEED)
rng = np.random.default_rng(BASE_SEED)

beam_sizes = [1, 3, 5, 7, 10] #12
n_instances = 50 #50
k_neighbors = 100
alpha = 0.01
mode_max_workers = 2
medoid_max_workers = 4

labels_stomach = [1, 5, 6, 7, 8]
labels_lung    = [0, 2, 3, 4, 9]

generic_symptoms_cols = [
    'fever_severity', 'cough_severity', 'chest_pain_severity',
    'abdominal_pain_severity', 'fatigue_level', 'nausea'
]

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
xgb_cl = xgb.XGBClassifier()
xgb_cl.load_model("xgb_model.json")
xgb_cl.set_params(n_jobs=1)
class_names = xgb_cl.classes_
qhat = compute_conformal_threshold(xgb_cl, X_conf_pred, y_conf_pred, alpha=alpha)

# =========================================================
# HELPER FUNCTIONS FROM YOUR UNION-PRUNING PIPELINE
# =========================================================

# ---------------------------------------------------------
# Helper: check if an anchor applies to a given instance
# (used inside union-pruning coverage computation)
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
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    ranges = np.ones(X.shape[1], dtype=float)
    symptom_range_width = float(symptom_range[1] - symptom_range[0])
    for j, fname in enumerate(feature_names):
        if fname in symptom_cols:
            ranges[j] = symptom_range_width if symptom_range_width > 0 else 1.0
        elif fname in TEST_BOUNDS:
            lo, hi = TEST_BOUNDS[fname]
            width = float(hi - lo)
            ranges[j] = width if width > 0 else 1.0

    valid = ~np.isnan(X)
    diff = np.abs(X[:, None, :] - X[None, :, :])
    valid_pair = valid[:, None, :] & valid[None, :, :]
    norm_diff = np.where(valid_pair, diff / ranges, 0.0)
    den = valid_pair.sum(axis=2)
    distances = np.ones((n, n), dtype=float)
    np.divide(norm_diff.sum(axis=2), den, out=distances, where=den > 0)
    return distances


def parse_anchor_condition(cond):
    cond = cond.strip()
    if "≤" in cond:
        feature, thresh = cond.split("≤", 1); op = "leq"
    elif "<=" in cond:
        feature, thresh = cond.split("<=", 1); op = "leq"
    elif ">" in cond:
        feature, thresh = cond.split(">", 1); op = "gt"
    elif "=" in cond:
        feature, thresh = cond.split("=", 1); op = "eq"
    else:
        return None

    feature = feature.strip()
    thresh = thresh.strip()
    try:
        return feature, op, float(thresh), True
    except ValueError:
        return feature, op, thresh, False


def anchor_mask_numpy(predicate_names, X_values, feature_to_idx):
    mask = np.ones(X_values.shape[0], dtype=bool)
    for cond in predicate_names:
        parsed = parse_anchor_condition(cond)
        if parsed is None:
            return np.zeros(X_values.shape[0], dtype=bool)

        feature, op, thresh, numeric = parsed
        col_idx = feature_to_idx.get(feature)
        if col_idx is None:
            return np.zeros(X_values.shape[0], dtype=bool)

        col = X_values[:, col_idx]
        valid = ~np.isnan(col)
        if op == "leq":
            if not numeric:
                return np.zeros(X_values.shape[0], dtype=bool)
            cond_mask = valid & (col <= thresh)
        elif op == "gt":
            if not numeric:
                return np.zeros(X_values.shape[0], dtype=bool)
            cond_mask = valid & (col > thresh)
        elif op == "eq":
            if numeric:
                cond_mask = valid & (np.abs(col - thresh) < 1e-20)
            else:
                cond_mask = valid & (col.astype(str) == str(thresh))
        else:
            return np.zeros(X_values.shape[0], dtype=bool)

        mask &= cond_mask
        if not mask.any():
            break
    return mask


def compute_anchor_coverage_masks(anchors, coverage_df, n_samples):
    sampled_idx = np.random.choice(coverage_df.shape[0], size=n_samples, replace=True)
    coverage_values = coverage_df.iloc[sampled_idx].to_numpy()
    feature_to_idx = {name: i for i, name in enumerate(coverage_df.columns)}

    masks = []
    for a in anchors:
        mask = anchor_mask_numpy(a['names'], coverage_values, feature_to_idx)
        masks.append(mask)
        a['cov_train'] = float(mask.mean())
    return masks


def dedupe_anchor_candidates(anchors):
    deduped = []
    seen = set()
    for anchor in anchors:
        key = tuple(anchor['names'])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(anchor)
    return deduped


# ---------------------------------------------------------
# UNION PRUNING – MODE 1 (max marginal gain each step)
# ---------------------------------------------------------
def union_prune_anchors_mode1(anchors, coverage_df, n_samples=10000, flatten_tol=1e-6):
    if not anchors:
        return [], 0.0, [], []

    n_universe = coverage_df.shape[0]
    if n_universe == 0:
        return [], 0.0, [], []

    masks = compute_anchor_coverage_masks(anchors, coverage_df, n_samples)
    n_cov = masks[0].shape[0]

    final_anchors = []
    covered_union = np.zeros(n_cov, dtype=bool)
    last_cumulative_coverage = 0.0

    remaining = set(range(len(anchors)))
    coverage_trajectory = []
    gain_trajectory = []

    while remaining:
        best_t = None
        best_gain = 0.0
        best_candidate_union = None

        for t in remaining:
            candidate_union = covered_union | masks[t]
            cumulative_coverage = float(candidate_union.mean())
            gain = cumulative_coverage - last_cumulative_coverage

            if gain > best_gain:
                best_gain = gain
                best_t = t
                best_candidate_union = candidate_union

        if best_t is None or best_gain <= flatten_tol:
            break

        final_anchors.append(anchors[best_t])
        covered_union = best_candidate_union
        last_cumulative_coverage = float(covered_union.mean())
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

    masks = compute_anchor_coverage_masks(anchors, coverage_df, n_samples)
    n_cov = masks[0].shape[0]

    sorted_t = sorted(range(len(anchors)), key=lambda t: anchors[t]['cov_train'], reverse=True)

    final_anchors = []
    covered_union = np.zeros(n_cov, dtype=bool)
    last_cumulative_coverage = 0.0

    coverage_trajectory = []
    gain_trajectory = []

    for rank, t in enumerate(sorted_t, start=1):
        candidate_union = covered_union | masks[t]
        cumulative_coverage = float(candidate_union.mean())

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


# ---------------------------------------------------------
# Helper: pick one label at random in same disease group (≠ true)
# ---------------------------------------------------------
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


def task_seed(*parts):
    seed_seq = np.random.SeedSequence([BASE_SEED, *map(int, parts)])
    return int(seed_seq.generate_state(1)[0])


def run_anchor_search_worker(payload):
    np.random.seed(int(payload['seed']))

    explainer = AnchorTabularExplainer(
        class_names=class_names,
        feature_names=payload['feature_names'],
        train_data=payload['train_data'],
        discretizer=None,
        categorical_names={},
    )
    t0 = time.perf_counter()
    exp, valid = explainer.explain_instance(
        payload['instance'],
        xgb_cl,
        mode="conformal",
        query_label=payload['label_to_exclude'],
        qhat=payload['qhat'],
        threshold=0.95,
        delta=0.01,
        tau=0.15,
        beam_size=payload['beam_size'],
        predicate_mode=payload['predicate_mode'],
        mean_instances=None
    )
    runtime = float(time.perf_counter() - t0)

    if payload['kind'] == 'original':
        cumulative_cov = exp.cumulative_coverage()
        if isinstance(cumulative_cov, (list, tuple, np.ndarray)):
            cumulative_cov = cumulative_cov[-1]
        return {
            'kind': 'original',
            'beam_size': int(payload['beam_size']),
            'runtime': runtime,
            'cumulative_coverage': float(cumulative_cov),
        }

    anchors = []
    m_id = int(payload['medoid_id'])
    for va in valid:
        anchors.append({
            'names': va['names'],
            'medoid_id': m_id,
            'source': 'valid_anchor',
            'precision': float(va['precision'][-1]),
            'coverage': float(va['coverage'][-1]),
        })

    main_names = exp.names()
    if not any(a['names'] == main_names for a in anchors):
        anchors.append({
            'names': main_names,
            'medoid_id': m_id,
            'source': 'main_anchor',
            'precision': float(exp.precision()),
            'coverage': float(exp.coverage()),
        })

    return {
        'kind': 'medoid',
        'beam_size': int(payload['beam_size']),
        'medoid_id': m_id,
        'runtime': runtime,
        'anchors': anchors,
    }


# =========================================================
# STORAGE FOR RESULTS
# =========================================================
coverage_by_beam_orig = {b: [] for b in beam_sizes}
runtime_by_beam_orig  = {b: [] for b in beam_sizes}

coverage_by_beam_u1 = {b: [] for b in beam_sizes}   # union mode 1: union coverage
runtime_by_beam_u1  = {b: [] for b in beam_sizes}

coverage_by_beam_u2 = {b: [] for b in beam_sizes}   # union mode 2: union coverage
runtime_by_beam_u2  = {b: [] for b in beam_sizes}

skipped_instances = 0

# ---------------------------------------------------------
# Select instances
# ---------------------------------------------------------
n_instances = min(n_instances, len(X_anchors))
selected_anchor_indices = np.random.choice(len(X_anchors), size=n_instances, replace=False)

# ---------------------------------------------------------
# OUTPUT FILE
# ---------------------------------------------------------
output_path = "beam_size_sensitivity_coverage_runtime_all_modes_optimzed.txt"
anchor_pool = ProcessPoolExecutor(
    max_workers=max(1, mode_max_workers + medoid_max_workers),
    mp_context=get_context("fork")
)

with open(output_path, "w") as f:
    f.write("BEAM SIZE SENSITIVITY – ORIGINAL MODE + MEDOID UNION PRUNING MODES 1 & 2\n")
    f.write("Coverage vs beam size (union coverage) + Runtime vs beam size\n")
    f.write(f"Number of anchor instances: {n_instances}\n")
    f.write(f"k (neighbors): {k_neighbors}\n")
    f.write(f"alpha: {alpha}\n")
    f.write(f"qhat: {qhat:.5f}\n")
    f.write(f"Beam sizes: {beam_sizes}\n")
    f.write(f"Mode workers: {mode_max_workers}\n")
    f.write(f"Medoid workers: {medoid_max_workers}\n")
    f.write("Label-to-exclude: ONE random label in the same disease group (≠ true label)\n")
    f.write("Anchor construction: medoid-based neighborhood + union pruning\n")
    f.write("=" * 100 + "\n\n")

    for run_id, anchor_idx in enumerate(selected_anchor_indices, 1):
        f.write("=" * 80 + "\n")
        f.write(f"Instance {run_id}/{n_instances} (anchor index = {anchor_idx})\n")

        # Target patient (no masking here)
        new_patient = X_anchors.iloc[anchor_idx].copy()
        true_label  = int(y_anchors.iloc[anchor_idx])

        anchor_pool_mask = np.ones(len(X_anchors), dtype=bool)
        anchor_pool_mask[anchor_idx] = False

        X_anchor_pool = X_anchors.iloc[anchor_pool_mask].reset_index(drop=True)
        y_anchor_pool = y_anchors.iloc[anchor_pool_mask].reset_index(drop=True)
        X_anchor_orig_pool = X_anchors_orig.iloc[anchor_pool_mask].reset_index(drop=True)

        # Choose label-to-exclude (random same group)
        label_to_exclude = pick_same_group_label_to_exclude_random(true_label, rng)
        if label_to_exclude is None:
            f.write("  -> No valid same-group label to exclude. Skipping instance.\n\n")
            skipped_instances += 1
            continue

        # Build 1-row DF for get_knn_subsets
        new_patient_df = new_patient.to_frame().T
        new_patient_df.index = [anchor_idx]

        # kNN neighborhood (binned + raw indices)
        subsets_list, y_subsets_list, similarities_list, indices_neighbors_list = get_knn_subsets(
            new_patient_df, X_anchor_pool, y_anchor_pool, k=k_neighbors
        )
        subset_new_patient = subsets_list[0]                 # binned neighbors (DataFrame)
        y_subset_new_patient = y_subsets_list[0]             # neighbor labels
        indices_neighbors = indices_neighbors_list[0]        # indices into *_orig pools

        subset_new_patient_raw = X_anchor_orig_pool.iloc[indices_neighbors].reset_index(drop=True)

        # conformal prediction sets on neighbors with cached qhat
        prediction_sets = construct_prediction_sets(xgb_cl, subset_new_patient, qhat)

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

        neighbors_excluding_target      = subset_new_patient.iloc[idxs_excluding].reset_index(drop=True)
        neighbors_excluding_target_raw  = subset_new_patient_raw.iloc[idxs_excluding].reset_index(drop=True)
        neighbors_labels = y_subset_new_patient.iloc[idxs_excluding].reset_index(drop=True)

        unique_labels = np.unique(neighbors_labels)
        f.write(f"  True label: {true_label} ({categories[true_label]})\n")
        f.write(f"  Label to exclude (same group): {label_to_exclude} ({categories[label_to_exclude]})\n")
        f.write(f"  qhat = {qhat:.5f} | #neighbors excluding label = {len(idxs_excluding)}\n")
        f.write(f"  Unique neighbor labels (indices): {unique_labels.tolist()}\n")
        f.write("  Unique neighbor labels (diseases): " +
                ", ".join(f"{int(l)} ({categories[int(l)]})" for l in unique_labels) + "\n\n")

        # -----------------------------------------------------
        # K-MEDOIDS PREPARATION (RAW GOWER) – same as big code
        # -----------------------------------------------------
        X_cluster_raw_df = neighbors_excluding_target_raw.reset_index(drop=True)
        X_cluster_raw    = X_cluster_raw_df.to_numpy(dtype=float)
        feature_names_raw = X_cluster_raw_df.columns.tolist()

        # Build label-specific clusters & indices
        cluster_list = []
        cluster_indices_list = []
        mean_instances_raw = []

        for lab in unique_labels:
            mask_lab = (neighbors_labels == lab)
            cluster = neighbors_excluding_target_raw.loc[mask_lab]
            cluster_indices = np.where(mask_lab.values)[0]
            cluster_list.append(cluster)
            cluster_indices_list.append(cluster_indices)
            mean_instances_raw.append(cluster.mean(axis=0))

        mean_instances_raw = np.vstack(mean_instances_raw)

        # Full distance matrix
        D = compute_gower_distance_matrix_raw(
            X_cluster_raw,
            feature_names=feature_names_raw,
            symptom_cols=generic_symptoms_cols,
            TEST_BOUNDS=TEST_BOUNDS,
            symptom_range=(0, 10)
        )

        # Snap each label-mean to nearest actual point -> init medoids
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
        f.write(f"  K-Medoids clustering with k={k} (unique labels among neighbors)\n")

        init_matrix = D[init_medoids, :]

        kmedoids = KMedoids(
            n_clusters=k,
            metric="precomputed",
            init=init_matrix,
            max_iter=300,
            random_state=0
        )
        kmedoids.fit(D)

        medoids_idx        = kmedoids.medoid_indices_
        cluster_labels_km  = kmedoids.labels_
        f.write(f"  Final medoids indices (within neighbors_excluding_target): {medoids_idx.tolist()}\n\n")

        # Medoids in raw space
        final_medoids_raw_df = X_cluster_raw_df.iloc[medoids_idx].copy().reset_index(drop=True)

        # Bin medoids to explainer feature space
        final_medoids_binned_df = bin_dataset(
            final_medoids_raw_df,
            TESTS=TESTS,
            generic_symptoms_cols=generic_symptoms_cols,
            verbose=False
        )
        # Overwrite generic symptoms with patient's observed values
        final_medoids_binned_df[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
        final_medoids_binned = final_medoids_binned_df.to_numpy()

        feature_cols = X_train.columns.tolist()
        medoid_train_data = subset_new_patient.to_numpy()

        # coverage_df for union pruning: use neighbor subset (binned)
        coverage_df = subset_new_patient

        # -----------------------------------------------------
        # LOOP OVER BEAM SIZES – UNION MODE 1 & 2
        # -----------------------------------------------------
        flatten_tol = 1e-4

        idx_example_to_anchor = int(rng.choice(idxs_excluding))
        anchor_row = subset_new_patient.iloc[idx_example_to_anchor].copy()

        anchor_row[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
        anchor_instance = anchor_row.to_numpy()

        indices_train_orig = [i for i in range(len(subset_new_patient)) if i != idx_example_to_anchor]

        orig_train_data = subset_new_patient.iloc[indices_train_orig].to_numpy()

        all_beam_tasks = []
        for b in beam_sizes:
            all_beam_tasks.append({
                'seed': task_seed(anchor_idx, b, 10, 0),
                'kind': 'original',
                'feature_names': feature_cols,
                'train_data': orig_train_data,
                'instance': anchor_instance,
                'label_to_exclude': int(label_to_exclude),
                'qhat': float(qhat),
                'beam_size': int(b),
                'predicate_mode': 'original',
            })

            for m_id, medoid_instance in enumerate(final_medoids_binned, 1):
                all_beam_tasks.append({
                    'seed': task_seed(anchor_idx, b, 30, m_id),
                    'kind': 'medoid',
                    'medoid_id': int(m_id),
                    'feature_names': feature_cols,
                    'train_data': medoid_train_data,
                    'instance': medoid_instance,
                    'label_to_exclude': int(label_to_exclude),
                    'qhat': float(qhat),
                    'beam_size': int(b),
                    'predicate_mode': 'medoid',
                })

        beam_outputs = {
            b: {'original': None, 'medoids': []}
            for b in beam_sizes
        }
        for output in anchor_pool.map(run_anchor_search_worker, all_beam_tasks):
            b = int(output['beam_size'])
            if output['kind'] == 'original':
                beam_outputs[b]['original'] = output
            else:
                beam_outputs[b]['medoids'].append(output)

        for b in beam_sizes:
            f.write(f"  >> Beam size B = {b}\n")
            orig_output = beam_outputs[b]['original']
            medoid_outputs = beam_outputs[b]['medoids']

            all_medoid_anchors = []
            for output in medoid_outputs:
                all_medoid_anchors.extend(output['anchors'])
            all_medoid_anchors = dedupe_anchor_candidates(all_medoid_anchors)

            medoid_search_runtime = max(
                (float(output['runtime']) for output in medoid_outputs),
                default=0.0
            )

            # =======================
            # UNION MODE 1
            # =======================
            t0 = time.perf_counter()
            final_anchors_u1, final_union_cov1, cov_traj_u1, gain_traj_u1 = (
                union_prune_anchors_mode1(
                    [anchor.copy() for anchor in all_medoid_anchors],
                    coverage_df,
                    n_samples=2500,
                    flatten_tol=flatten_tol
                )
            )
            t1 = time.perf_counter()
            rt_u1 = float(medoid_search_runtime + (t1 - t0))

            # store union coverage (not just first anchor coverage)
            coverage_by_beam_u1[b].append(float(final_union_cov1))
            runtime_by_beam_u1[b].append(rt_u1)

            f.write(f"    [union1] union_coverage={final_union_cov1:.5f} | runtime={rt_u1:.4f}s\n")

            # =======================
            # UNION MODE 2
            # =======================
            t0 = time.perf_counter()
            final_anchors_u2, final_union_cov2, cov_traj_u2, gain_traj_u2 = (
                union_prune_anchors_mode2(
                    [anchor.copy() for anchor in all_medoid_anchors],
                    coverage_df,
                    n_samples=2500,
                    flatten_tol=flatten_tol
                )
            )
            t1 = time.perf_counter()
            rt_u2 = float(medoid_search_runtime + (t1 - t0))

            coverage_by_beam_u2[b].append(float(final_union_cov2))
            runtime_by_beam_u2[b].append(rt_u2)

            f.write(f"    [union2] union_coverage={final_union_cov2:.5f} | runtime={rt_u2:.4f}s\n")

            # =======================
            # ORIGINAL MODE
            # =======================
            cumulative_cov_orig = float(orig_output['cumulative_coverage'])
            rt_orig = float(orig_output['runtime'])

            coverage_by_beam_orig[b].append(float(cumulative_cov_orig))
            runtime_by_beam_orig[b].append(rt_orig)

            f.write(
                f"    [original] cumulative_coverage={float(cumulative_cov_orig):.5f} | "
                f"runtime={rt_orig:.4f}s\n"
            )
        f.write("\n")

    # ---------------------------------------------------------
    # AGGREGATE STATISTICS
    # ---------------------------------------------------------
    def mean_std(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.nan, np.nan
        return float(np.mean(arr)), float(np.std(arr))

    mean_cov_u1, std_cov_u1, mean_rt_u1, std_rt_u1 = [], [], [], []
    mean_cov_u2, std_cov_u2, mean_rt_u2, std_rt_u2 = [], [], [], []
    mean_cov_orig, std_cov_orig, mean_rt_orig, std_rt_orig = [], [], [], []

    f.write("\n" + "=" * 100 + "\n")
    f.write("AGGREGATE STATISTICS ACROSS INSTANCES (UNION COVERAGE)\n")
    f.write("=" * 100 + "\n\n")

    f.write(f"Used {n_instances - skipped_instances} instances; skipped {skipped_instances}.\n\n")

    f.write("UNION MODE 1:\n")
    for b in beam_sizes:
        m_cov, s_cov = mean_std(coverage_by_beam_u1[b])
        m_rt, s_rt   = mean_std(runtime_by_beam_u1[b])

        mean_cov_u1.append(m_cov)
        std_cov_u1.append(s_cov)
        mean_rt_u1.append(m_rt)
        std_rt_u1.append(s_rt)

        f.write(
            f"  Beam {b:>2}: mean union_coverage = {m_cov:.5f} (std={s_cov:.5f}), "
            f"mean runtime = {m_rt:.5f}s (std={s_rt:.5f}s)\n"
        )

    f.write("\nUNION MODE 2:\n")
    for b in beam_sizes:
        m_cov, s_cov = mean_std(coverage_by_beam_u2[b])
        m_rt, s_rt   = mean_std(runtime_by_beam_u2[b])

        mean_cov_u2.append(m_cov)
        std_cov_u2.append(s_cov)
        mean_rt_u2.append(m_rt)
        std_rt_u2.append(s_rt)

        f.write(
            f"  Beam {b:>2}: mean union_coverage = {m_cov:.5f} (std={s_cov:.5f}), "
            f"mean runtime = {m_rt:.5f}s (std={s_rt:.5f}s)\n"
        )
    f.write("\nORIGINAL MODE:\n")
    for b in beam_sizes:
        m_cov, s_cov = mean_std(coverage_by_beam_orig[b])
        m_rt, s_rt   = mean_std(runtime_by_beam_orig[b])
        mean_cov_orig.append(m_cov)
        std_cov_orig.append(s_cov)
        mean_rt_orig.append(m_rt)
        std_rt_orig.append(s_rt)

        f.write(
            f"  Beam {b:>2}: mean union_coverage = {m_cov:.5f} (std={s_cov:.5f}), "
            f"mean runtime = {m_rt:.5f}s (std={s_rt:.5f}s)\n"
        )


anchor_pool.shutdown()
print(f"Done. Full report saved to: {output_path}")

# =========================================================
# PLOTS (union coverage vs B, runtime vs B, trade-off)
# =========================================================

def plot_coverage_vs_beam(coverage_by_beam, mean_cov, mode_name, pdf_name):
    fig, ax = plt.subplots(figsize=(7, 5))

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
            sns.lineplot(
                x=beam_sizes,
                y=y,
                ax=ax,
                color=COLORBLIND_PALETTE[0],
                alpha=0.2,
                legend=False,
            )

    sns.lineplot(
        x=beam_sizes,
        y=mean_cov,
        ax=ax,
        marker='o',
        linewidth=2,
        color=COLORBLIND_PALETTE[1],
        label='Mean union coverage',
    )

    ax.set_xlim(1, 10)
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylim(0, 1.0)

    ax.set_xlabel("Beam size B")
    ax.set_ylabel("Union coverage")
    # plt.title(f"Union coverage vs Beam size ({mode_name})")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdf_name)
    plt.show()
    
def plot_coverage_vs_beam_CI(coverage_by_beam, mode_name, pdf_name):
    fig, ax = plt.subplots(figsize=(7, 5))

    means = []
    ci_lows = []
    ci_highs = []

    for b in beam_sizes:
        vals = np.asarray(coverage_by_beam[b], dtype=float)
        vals = vals[~np.isnan(vals)]

        mean = vals.mean()

        if len(vals) > 1:
            sem = stats.sem(vals)
            low, high = stats.t.interval(
                confidence=0.95,
                df=len(vals) - 1,
                loc=mean,
                scale=sem,
            )
        else:
            low, high = mean, mean

        means.append(mean)
        ci_lows.append(low)
        ci_highs.append(high)

    means = np.asarray(means)
    ci_lows = np.asarray(ci_lows)
    ci_highs = np.asarray(ci_highs)

    # Since coverage is bounded in [0, 1]
    ci_lows = np.clip(ci_lows, 0, 1)
    ci_highs = np.clip(ci_highs, 0, 1)

    sns.lineplot(
        x=beam_sizes,
        y=means,
        ax=ax,
        marker="o",
        linewidth=2,
        color=COLORBLIND_PALETTE[1],
        label="Mean union coverage",
    )

    ax.fill_between(
        beam_sizes,
        ci_lows,
        ci_highs,
        color=COLORBLIND_PALETTE[1],
        alpha=0.25,
        label="95% CI",
    )

    ax.set_xlim(1, 10)
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylim(0, 1.0)

    ax.set_xlabel("Beam size")
    ax.set_ylabel("Union coverage")
    # plt.title(rf"Union coverage vs Beam size ({mode_name})")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdf_name, bbox_inches="tight")
    plt.show()


def plot_runtime_vs_beam(runtime_by_beam, mean_rt, mode_name, pdf_name):
    fig, ax = plt.subplots(figsize=(7, 5))

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
            sns.lineplot(
                x=beam_sizes,
                y=y,
                ax=ax,
                color=COLORBLIND_PALETTE[2],
                alpha=0.2,
                legend=False,
            )

    sns.lineplot(
        x=beam_sizes,
        y=mean_rt,
        ax=ax,
        marker='o',
        linewidth=2,
        color=COLORBLIND_PALETTE[3],
        label='Mean runtime',
    )

    ax.set_xlim(1, 10)
    ax.set_xticks(np.arange(1, 11, 1))
    # plt.yticks(np.arange(0, 21, 1))
    ax.set_ylim(0, 120)

    ax.set_xlabel("Beam size")
    ax.set_ylabel("Runtime per explanation (seconds)")
    # plt.title(f"Runtime vs Beam size ({mode_name})")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdf_name)
    plt.show()

def plot_runtime_vs_beam_CI(runtime_by_beam, mode_name, pdf_name):
    fig, ax = plt.subplots(figsize=(7, 5))

    means = []
    ci_lows = []
    ci_highs = []

    for b in beam_sizes:
        vals = np.asarray(runtime_by_beam[b], dtype=float)
        vals = vals[~np.isnan(vals)]

        mean = vals.mean()
        sem = stats.sem(vals)  # standard error of the mean

        if len(vals) > 1:
            ci = stats.t.interval(
                confidence=0.95,
                df=len(vals) - 1,
                loc=mean,
                scale=sem,
            )
            low, high = ci
        else:
            low, high = mean, mean

        means.append(mean)
        ci_lows.append(low)
        ci_highs.append(high)

    means = np.asarray(means)
    ci_lows = np.asarray(ci_lows)
    ci_highs = np.asarray(ci_highs)

    sns.lineplot(
        x=beam_sizes,
        y=means,
        ax=ax,
        marker="o",
        linewidth=2,
        color=COLORBLIND_PALETTE[3],
        label="Mean runtime",
    )

    ax.fill_between(
        beam_sizes,
        ci_lows,
        ci_highs,
        color=COLORBLIND_PALETTE[3],
        alpha=0.25,
        label="95% CI",
    )

    ax.set_xlim(1, 10)
    ax.set_xticks(np.arange(1, 11, 1))
    ax.set_ylim(0, 120)

    ax.set_xlabel("Beam size")
    ax.set_ylabel("Runtime per explanation (seconds)")
    # plt.title(rf"Runtime vs Beam size ({mode_name})")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdf_name, bbox_inches="tight")
    plt.show()


def plot_tradeoff(mean_cov, mean_rt, mode_name, pdf_name):
    fig, ax = plt.subplots(figsize=(7, 5))

    mean_cov_arr = np.array(mean_cov, dtype=float)
    mean_rt_arr  = np.array(mean_rt, dtype=float)
    beam_arr     = np.array(beam_sizes, dtype=int)

    sns.lineplot(
        x=mean_rt_arr,
        y=mean_cov_arr,
        ax=ax,
        marker='o',
        linewidth=2,
        color=COLORBLIND_PALETTE[4],
    )

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

        ax.annotate(
            f"B={b}",
            (x, y),
            textcoords="offset points",
            xytext=offset,
            ha='left',
            va=va,
            fontsize=9
        )

    # plt.yticks(np.arange(0, 1.01, 0.1))
    # plt.ylim(0, 1.0)
    # plt.xticks(np.arange(1, 13, 1))

    ax.set_xlabel("Mean runtime per explanation (seconds)")
    ax.set_ylabel("Mean union coverage (sampled)")
    # plt.title(f"Coverage–runtime trade-off across beam sizes ({mode_name})", pad=20)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(pdf_name)
    plt.show()


# Build mean curves
mean_cov_u1_arr = [float(np.mean(coverage_by_beam_u1[b])) if len(coverage_by_beam_u1[b]) > 0 else np.nan
                   for b in beam_sizes]
mean_rt_u1_arr  = [float(np.mean(runtime_by_beam_u1[b]))  if len(runtime_by_beam_u1[b])  > 0 else np.nan
                   for b in beam_sizes]
mean_cov_u2_arr = [float(np.mean(coverage_by_beam_u2[b])) if len(coverage_by_beam_u2[b]) > 0 else np.nan
                   for b in beam_sizes]
mean_rt_u2_arr  = [float(np.mean(runtime_by_beam_u2[b]))  if len(runtime_by_beam_u2[b])  > 0 else np.nan
                   for b in beam_sizes]
mean_cov_orig_arr = [
    float(np.mean(coverage_by_beam_orig[b]))
    if len(coverage_by_beam_orig[b]) > 0 else np.nan
    for b in beam_sizes
]

mean_rt_orig_arr = [
    float(np.mean(runtime_by_beam_orig[b]))
    if len(runtime_by_beam_orig[b]) > 0 else np.nan
    for b in beam_sizes
]

# PLOTS FOR UNION MODE 1
# plot_coverage_vs_beam(
#     coverage_by_beam_u1,
#     mean_cov_u1_arr,
#     mode_name="$\textsc{MG}$",
#     pdf_name="union_coverage_vs_beamsize_union1_medoid.pdf"
# )

plot_coverage_vs_beam_CI(
    coverage_by_beam_u1,    
    mode_name="$\textsc{MG}$",
    pdf_name="union_coverage_vs_beamsize_union1_medoid.pdf"
)

# plot_runtime_vs_beam(
#     runtime_by_beam_u1,
#     mean_rt_u1_arr,
#     mode_name=r"$\textsc{MG}$",
#     pdf_name="runtime_vs_beamsize_union1_medoid.pdf"
# )

plot_runtime_vs_beam_CI(
    runtime_by_beam_u1,    
    mode_name=r"$\textsc{MG}$",
    pdf_name="runtime_vs_beamsize_union1_medoid.pdf"
)

plot_tradeoff(
    mean_cov_u1_arr,
    mean_rt_u1_arr,
    mode_name=r"$\textsc{MG}$",
    pdf_name="tradeoff_coverage_vs_runtime_union1_medoid.pdf"
)

# PLOTS FOR UNION MODE 2
# plot_coverage_vs_beam(
#     coverage_by_beam_u2,
#     mean_cov_u2_arr,
#     mode_name=r"$\textsc{CG}$",
#     pdf_name="union_coverage_vs_beamsize_union2_medoid.pdf"
# )

plot_coverage_vs_beam_CI(
    coverage_by_beam_u2,    
    mode_name=r"$\textsc{CG}$",
    pdf_name="union_coverage_vs_beamsize_union2_medoid.pdf"
)

# plot_runtime_vs_beam(
#     runtime_by_beam_u2,
#     mean_rt_u2_arr,
#     mode_name=r"$\textsc{CG}$",
#     pdf_name="runtime_vs_beamsize_union2_medoid.pdf"
# )

plot_runtime_vs_beam_CI(
    runtime_by_beam_u2,
    mode_name=r"$\textsc{CG}$",
    pdf_name="runtime_vs_beamsize_union2_medoid.pdf"
)

plot_tradeoff(
    mean_cov_u2_arr,
    mean_rt_u2_arr,
    mode_name=r"$\textsc{CG}$",
    pdf_name="tradeoff_coverage_vs_runtime_union2_medoid.pdf"
)

# PLOTS FOR ORIGINAL MODE
# plot_coverage_vs_beam(
#     coverage_by_beam_orig,
#     mean_cov_orig_arr,
#     mode_name=r"$\textsc{OS}$",
#     pdf_name="union_coverage_vs_beamsize_original_anchor_set.pdf"
# )

plot_coverage_vs_beam_CI(
    coverage_by_beam_orig,    
    mode_name=r"$\textsc{OS}$",
    pdf_name="union_coverage_vs_beamsize_original_anchor_set.pdf"
)

# plot_runtime_vs_beam(
#     runtime_by_beam_orig,
#     mean_rt_orig_arr,
#     mode_name=r"$\textsc{OS}$",
#     pdf_name="runtime_vs_beamsize_original_anchor_set.pdf"
# )

plot_runtime_vs_beam_CI(
    runtime_by_beam_orig,
    mode_name=r"$\textsc{OS}$",
    pdf_name="runtime_vs_beamsize_original_anchor_set.pdf"
)

plot_tradeoff(
    mean_cov_orig_arr,
    mean_rt_orig_arr,
    mode_name=r"$\textsc{OS}$",
    pdf_name="tradeoff_coverage_vs_runtime_original_anchor_set.pdf"
)
