# ---------------------------------------------------------
# Experiment: true-label exclusion under no additional missingness
# ---------------------------------------------------------
#
# This script evaluates the conformal-anchor framework in a stress-test setting
# where the target label to exclude is the true label of each test instance.
#
# For each selected test patient:
#   1. build a local neighborhood of k = 100 nearest anchor-set instances;
#   2. compute conformal prediction sets for the neighbors;
#   3. retain the neighbors whose prediction sets exclude the patient's true label;
#   4. generate conformal anchors explaining the exclusion of the true label;
#   5. compare five construction modes:
#        - original mode;
#        - medoid mode based on K-Medoids with raw Gower distance;
#        - union pruning mode 1, selecting anchors by maximum marginal union coverage gain;
#        - union pruning mode 2, selecting anchors by decreasing individual coverage.
#
# The experiment records, for each mode:
#   - main-anchor precision and coverage;
#   - cumulative / union coverage;
#   - average precision over all valid anchors;
#   - number of valid anchors and involved features;
#   - whether the main anchor or any valid/selected anchor applies to the
#     original patient.
#
# This run uses no additional missingness injection. Missing values already
# present in the patient are treated as not satisfying anchor predicates.
# ---------------------------------------------------------

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from anchor.anchor.anchor_tabular import AnchorTabularExplainer
from model_training import (
    X_train, X_conf_pred, y_conf_pred,
    X_anchors, y_anchors, X_test, y_test,
    X_anchors_orig, y_anchors_orig,
    categories
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
import pandas as pd
import numpy as np
import time

BASE_SEED = 1
np.random.seed(BASE_SEED)

# ---------------------------------------------------------
# Helper functions overview
# ---------------------------------------------------------
#
# anchor_applies_to_instance:
#   Checks whether all predicates of an anchor are satisfied by a patient.
#   Missing values are treated as predicate failures.
#
# gower_distance_to_all_raw:
#   Computes raw-space Gower distances between one point and a set of points.
#
# compute_gower_distance_matrix_raw:
#   Builds the pairwise raw-space Gower distance matrix used by K-Medoids.
#
# union_prune_anchors_mode1:
#   Greedy union-pruning strategy. At each step, selects the anchor with the
#   largest marginal increase in union coverage.
#
# union_prune_anchors_mode2:
#   Simpler union-pruning strategy. Sorts anchors by individual coverage and
#   keeps adding anchors while union coverage still increases enough.
#
# extract_feature_name:
#   Extracts the feature name from a predicate string.
#
# valid_anchor_stats:
#   Computes summary statistics over all valid anchors returned by the anchor
#   search procedure.
# ---------------------------------------------------------

def anchor_applies_to_instance(predicate_names, patient_series):
    for cond in predicate_names:
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


generic_symptoms_cols = [
    'fever_severity', 'cough_severity', 'chest_pain_severity',
    'abdominal_pain_severity', 'fatigue_level', 'nausea'
]

def gower_distance_to_all_raw(X, x, feature_names, symptom_cols, TEST_BOUNDS, symptom_range=(0, 10)):
    return 1.0 - gower_similarity_mixed_raw(
        anchor_vectors=X,
        test_vector=x,
        feature_names=feature_names,
        symptom_cols=symptom_cols,
        TEST_BOUNDS=TEST_BOUNDS,
        feature_range=symptom_range
    )

def compute_gower_distance_matrix_raw(X, feature_names, symptom_cols, TEST_BOUNDS, symptom_range=(0, 10)):
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
    return np.where(den > 0, norm_diff.sum(axis=2) / den, 1.0)


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
    rng = np.random.default_rng(123)
    sampled_idx = rng.choice(coverage_df.shape[0], size=n_samples, replace=True)
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
    last_cum = 0.0
    remaining = set(range(len(anchors)))
    cov_traj, gain_traj = [], []

    while remaining:
        best_t, best_gain, best_union = None, 0.0, None
        for t in remaining:
            candidate_union = covered_union | masks[t]
            cum = float(candidate_union.mean())
            gain = cum - last_cum
            if gain > best_gain:
                best_gain, best_t, best_union = gain, t, candidate_union

        if best_t is None or best_gain <= flatten_tol:
            break

        final_anchors.append(anchors[best_t])
        covered_union = best_union
        last_cum = float(covered_union.mean())
        remaining.remove(best_t)

        cov_traj.append(last_cum)
        gain_traj.append(best_gain)

    return final_anchors, last_cum, cov_traj, gain_traj


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
    last_cum = 0.0
    cov_traj, gain_traj = [], []

    for rank, t in enumerate(sorted_t, start=1):
        candidate_union = covered_union | masks[t]
        cum = float(candidate_union.mean())

        if rank == 1:
            final_anchors.append(anchors[t])
            covered_union = candidate_union
            last_cum = cum
            cov_traj.append(cum)
            gain_traj.append(cum)
        else:
            gain = cum - last_cum
            if gain <= flatten_tol:
                break
            final_anchors.append(anchors[t])
            covered_union = candidate_union
            last_cum = cum
            cov_traj.append(cum)
            gain_traj.append(gain)

    return final_anchors, last_cum, cov_traj, gain_traj


def extract_feature_name(cond_str):
    cond_str = cond_str.strip()
    for op in ["≤", "<=", ">", "="]:
        if op in cond_str:
            return cond_str.split(op, 1)[0].strip()
    return cond_str


def valid_anchor_stats(valid_anchors):
    """Return avg_feats, num_unique_feats, avg_precision_valid over VALID anchors."""
    if not valid_anchors:
        return 0.0, 0, np.nan
    avg_feats = float(np.mean([len(va['feature']) for va in valid_anchors]))
    unique_feats = set()
    for va in valid_anchors:
        unique_feats |= set(va['feature'])
    num_unique_feats = len(unique_feats)
    avg_prec_valid = float(np.mean([va['precision'][-1] for va in valid_anchors]))
    return avg_feats, num_unique_feats, avg_prec_valid


def task_seed(*parts):
    seed_seq = np.random.SeedSequence([BASE_SEED, *map(int, parts)])
    return int(seed_seq.generate_state(1)[0])


def run_anchor_search_worker(payload):
    np.random.seed(int(payload['seed']))

    explainer = AnchorTabularExplainer(
        class_names=class_names,
        feature_names=payload['feature_cols'],
        train_data=payload['train_data'],
        discretizer=None,
        categorical_names={},
    )

    exp, valid = explainer.explain_instance(
        payload['instance'],
        xgb_cl,
        mode="conformal",
        query_label=payload['label_to_exclude'],
        qhat=payload['qhat'],
        threshold=0.95,
        delta=payload['delta'],
        tau=0.15,
        batch_size=payload['anchor_batch_size'],
        coverage_samples=payload['anchor_coverage_samples'],
        beam_size=payload['beam_size'],
        predicate_mode=payload['predicate_mode'],
        mean_instances=payload.get('mean_instances'),
    )

    patient = payload['patient']
    avg_feats, uniq_feats, avg_prec_valid = valid_anchor_stats(valid)
    main_applies = int(anchor_applies_to_instance(exp.names(), patient))
    num_apply = int(sum(anchor_applies_to_instance(va['names'], patient) for va in valid))

    if payload['mode'] == 'medoid':
        m_id = int(payload['medoid_id'])
        lines = [
            f"\n--- Medoid {m_id}/{payload['n_medoids']} ---\n",
            f"Main anchor: {' AND '.join(exp.names())}\n",
            f"Main precision: {exp.precision():.3f}\n",
            f"Main coverage: {exp.coverage():.6f}\n",
            f"Cumulative coverage: {exp.cumulative_coverage():.6f}\n",
            f"Avg precision over ALL valid anchors: {avg_prec_valid}\n",
            f"Does MAIN apply to patient? {bool(main_applies)}\n",
            f"#valid anchors applying: {num_apply} / {len(valid)}\n",
        ]
        result = {
            'instance_idx': payload['new_patient_idx'],
            'mode': 'medoid',
            'medoid_id': m_id,
            'label_to_exclude': payload['label_to_exclude'],
            'main_precision': float(exp.precision()),
            'main_coverage': float(exp.coverage()),
            'cumulative_coverage': float(exp.cumulative_coverage()),
            'avg_precision_valid': float(avg_prec_valid) if not np.isnan(avg_prec_valid) else np.nan,
            'avg_feats_valid': float(avg_feats),
            'num_unique_feats_valid': int(uniq_feats),
            'num_valid_anchors': int(len(valid)),
            'main_applies': int(main_applies),
            'num_valid_apply': int(num_apply),
        }

        anchors = []
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

        return lines, [result], anchors

    lines = [
        f"{payload['title']}\n",
        f"Main anchor: {' AND '.join(exp.names())}\n",
        f"Main precision: {exp.precision():.3f}\n",
        f"Main coverage: {exp.coverage():.6f}\n",
        f"Cumulative coverage: {exp.cumulative_coverage():.6f}\n",
        f"Avg precision over ALL valid anchors: {avg_prec_valid}\n",
        f"Does MAIN apply to patient? {bool(main_applies)}\n",
        f"#valid anchors applying: {num_apply} / {len(valid)}\n\n",
    ]
    result = {
        'instance_idx': payload['new_patient_idx'],
        'mode': payload['mode'],
        'label_to_exclude': payload['label_to_exclude'],
        'main_precision': float(exp.precision()),
        'main_coverage': float(exp.coverage()),
        'cumulative_coverage': float(exp.cumulative_coverage()),
        'avg_precision_valid': float(avg_prec_valid) if not np.isnan(avg_prec_valid) else np.nan,
        'avg_feats_valid': float(avg_feats),
        'num_unique_feats_valid': int(uniq_feats),
        'num_valid_anchors': int(len(valid)),
        'main_applies': int(main_applies),
        'num_valid_apply': int(num_apply),
    }
    return lines, [result], []


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------
xgb_cl = xgb.XGBClassifier()
xgb_cl.load_model("xgb_model.json")
xgb_cl.set_params(n_jobs=1)
class_names = xgb_cl.classes_

# ---------------------------------------------------------
# EXPERIMENT SETTINGS
# ---------------------------------------------------------
start_time = time.time()

n_instances = 1000
n_instances = min(n_instances, len(X_test))
beam_size = 5 #10 #1
beam_size_medoid = 5 #10
alpha = 0.01
delta = 0.01
anchor_coverage_samples = 2500
anchor_batch_size = 100
union_samples = 2500
flatten_tol = 1e-2
mode_max_workers = 2
medoid_max_workers = 4

# Sample first, then compute kNN only for the patients that will be evaluated.
selected_test_indices = np.random.choice(len(X_test), size=n_instances, replace=False)
all_subsets, y_subsets, similarities_list, indices_neighbors = get_knn_subsets(
    X_test.iloc[selected_test_indices].reset_index(drop=True),
    X_anchors,
    y_anchors,
    k=100
)

# qhat is invariant across patients because the model, calibration set, and alpha
# do not change.
qhat = compute_conformal_threshold(xgb_cl, X_conf_pred, y_conf_pred, alpha=alpha)

results = []
output_path = (
    f"experiment_exclude_TRUE_label_all_modes_NO_missing_OPTIMIZED_"
    f"{beam_size}_{beam_size_medoid}_new_max_"
    f"delta{str(delta).replace('.', '_')}_"
    f"{n_instances}instances.txt"
)
anchor_pool = ProcessPoolExecutor(
    max_workers=max(1, mode_max_workers + medoid_max_workers),
    mp_context=get_context("fork")
)

with open(output_path, "w") as f:
    f.write("EXPERIMENT: Exclude TRUE label\n")
    f.write("Modes: original, medoid, union_mode1, union_mode2\n")
    f.write(f"Number of test instances: {n_instances}\n")
    f.write(f"Beam size: {beam_size}\n")
    f.write(f"Medoid beam size: {beam_size_medoid}\n")
    f.write(f"alpha: {alpha}\n")
    f.write(f"qhat: {qhat:.6f}\n")
    f.write(f"Anchor coverage samples: {anchor_coverage_samples}\n")
    f.write(f"Union pruning samples: {union_samples}\n")
    f.write(f"Mode workers: {mode_max_workers}\n")
    f.write(f"Medoid workers: {medoid_max_workers}\n")
    f.write("=" * 100 + "\n\n")

    counter_no_ps_excluding_true = 0

    for run_id, new_patient_idx in enumerate(selected_test_indices, 1):
        subset_pos = run_id - 1
        f.write(f"### INSTANCE {run_id}/{n_instances} (test index = {new_patient_idx})\n")
        f.write("-" * 80 + "\n")

        # -----------------------------
        # 1) patient info (NO missingness)
        # -----------------------------
        new_patient = X_test.iloc[new_patient_idx].copy()
        true_label = int(y_test.iloc[new_patient_idx])
        pred_label = int(xgb_cl.predict(new_patient.values.reshape(1, -1))[0])

        f.write(f"Patient: {new_patient.to_dict()}\n")

        f.write(f"True label: {true_label} ({categories[true_label]})\n")
        f.write(f"Pred label: {pred_label} ({categories[pred_label]})\n\n")

        # -----------------------------
        # 2) local neighborhood construction
        # -----------------------------
        subset_new_patient = all_subsets[subset_pos]
        y_subset_new_patient = y_subsets[subset_pos]

        subset_new_patient_raw = X_anchors_orig.iloc[
            indices_neighbors[subset_pos]
        ].reset_index(drop=True)
        y_subset_new_patient_raw = y_anchors_orig.iloc[
            indices_neighbors[subset_pos]
        ].reset_index(drop=True)

        # -----------------------------
        # 3) conformal prediction sets on neighbors
        # -----------------------------
        prediction_sets = construct_prediction_sets(xgb_cl, subset_new_patient, qhat)

        label_to_exclude = true_label
        f.write(f"qhat: {qhat:.6f}\n")
        f.write(f"Label to exclude (TRUE): {label_to_exclude} ({categories[label_to_exclude]})\n")

        label_col = int(np.where(class_names == label_to_exclude)[0][0])
        idxs_excl = np.flatnonzero(~prediction_sets[:, label_col]).tolist()

        if len(idxs_excl) == 0:
            f.write("No neighbors' prediction sets exclude the TRUE label. Skipping this instance.\n\n")
            counter_no_ps_excluding_true += 1
            continue

        neighbors_excl = subset_new_patient.iloc[idxs_excl]
        neighbors_excl_raw = subset_new_patient_raw.iloc[idxs_excl]
        neighbors_labels = y_subset_new_patient.iloc[idxs_excl]
        unique_labels = np.unique(neighbors_labels)

        f.write(f"#neighbors excluding TRUE label: {len(idxs_excl)}\n")
        f.write(f"Unique labels among those neighbors: {unique_labels.tolist()}\n\n")

        # -----------------------------
        # 4) anchor instance (picked among excluding-target neighbors)
        # -----------------------------
        idx_example_to_anchor = int(np.random.choice(idxs_excl))
        anchor_row = neighbors_excl.loc[idx_example_to_anchor].copy()

        # IMPORTANT: no missingness => patient has all generic symptoms,
        # we still overwrite for consistency with your pipeline
        anchor_row[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
        anchor_instance = anchor_row.to_numpy()

        # -----------------------------
        # 5) raw per-label means, used only to initialize K-Medoids
        # -----------------------------
        mean_instances_raw = []
        cluster_indices_list = []
        for lab in unique_labels:
            mask_lab = (neighbors_labels == lab)
            cluster = neighbors_excl_raw.loc[mask_lab]
            cluster_indices = np.where(mask_lab.values)[0]
            cluster_indices_list.append(cluster_indices)
            mean_instances_raw.append(cluster.mean(axis=0))
        mean_instances_raw = np.vstack(mean_instances_raw)

        # -----------------------------
        # 6) K-Medoids (RAW Gower)
        # -----------------------------
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

        init_medoids = []
        for i, m in enumerate(mean_instances_raw):
            X_cluster_label = X_cluster_raw[cluster_indices_list[i]]
            d_to_all = gower_distance_to_all_raw(
                X_cluster_label, m,
                feature_names=feature_names_raw,
                symptom_cols=generic_symptoms_cols,
                TEST_BOUNDS=TEST_BOUNDS,
                symptom_range=(0, 10)
            )
            local_idx = int(np.argmin(d_to_all))
            global_idx = int(cluster_indices_list[i][local_idx])
            init_medoids.append(global_idx)

        k = len(init_medoids)
        init_matrix = D[init_medoids, :]
        kmedoids = KMedoids(
            n_clusters=k, metric="precomputed", init=init_matrix, max_iter=300, random_state=0
        )
        kmedoids.fit(D)
        medoids_idx = kmedoids.medoid_indices_

        final_medoids_raw_df = X_cluster_raw_df.iloc[medoids_idx].copy().reset_index(drop=True)
        final_medoids_binned_df = bin_dataset(
            final_medoids_raw_df, TESTS=TESTS, generic_symptoms_cols=generic_symptoms_cols, verbose=False
        )
        final_medoids_binned_df[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
        final_medoids_binned = final_medoids_binned_df.to_numpy()

        # -----------------------------
        # 7) explainers
        # -----------------------------
        feature_cols = X_train.columns.tolist()
        categorical_names = {}
        indices_train_orig = [i for i in range(len(subset_new_patient)) if i != idx_example_to_anchor]

        orig_train_data = subset_new_patient.iloc[indices_train_orig].to_numpy()
        medoid_train_data = subset_new_patient.to_numpy()

        common_payload = {
            'feature_cols': feature_cols,
            'patient': new_patient,
            'new_patient_idx': int(new_patient_idx),
            'label_to_exclude': int(label_to_exclude),
            'qhat': float(qhat),
            'delta': float(delta),
            'anchor_batch_size': int(anchor_batch_size),
            'anchor_coverage_samples': int(anchor_coverage_samples),
        }
        anchor_tasks = [
            {
                **common_payload,
                'seed': task_seed(new_patient_idx, 10, 0),
                'title': 'ORIGINAL MODE',
                'mode': 'original',
                'train_data': orig_train_data,
                'instance': anchor_instance,
                'beam_size': int(beam_size),
                'predicate_mode': 'original',
                'mean_instances': None,
            },
        ]
        for m_id, medoid_instance in enumerate(final_medoids_binned, 1):
            anchor_tasks.append({
                **common_payload,
                'seed': task_seed(new_patient_idx, 30, m_id),
                'mode': 'medoid',
                'train_data': medoid_train_data,
                'instance': medoid_instance,
                'beam_size': int(beam_size_medoid),
                'predicate_mode': 'medoid',
                'mean_instances': None,
                'medoid_id': int(m_id),
                'n_medoids': int(len(final_medoids_binned)),
            })

        anchor_payloads = list(anchor_pool.map(run_anchor_search_worker, anchor_tasks))

        original_lines, original_results, _ = anchor_payloads[0]
        medoid_payloads = anchor_payloads[1:]

        f.writelines(original_lines)
        f.write("MEDOID MODE (KMedoids + RAW Gower)\n")
        results.extend(original_results)

        all_medoid_anchors_for_instance = []
        for medoid_lines, medoid_results, medoid_anchors in medoid_payloads:
            f.writelines(medoid_lines)
            results.extend(medoid_results)
            all_medoid_anchors_for_instance.extend(medoid_anchors)
        f.write("\n")

        all_medoid_anchors_for_instance = dedupe_anchor_candidates(
            all_medoid_anchors_for_instance
        )

        # =====================================================
        # PRE-PRUNING: does ANY candidate medoid anchor apply to the patient?
        # (candidate pool passed to both union pruning modes)
        # =====================================================
        any_apply_before_union = int(any(
            anchor_applies_to_instance(a['names'], new_patient)
            for a in all_medoid_anchors_for_instance
        ))

        f.write(f"Pre-pruning: any candidate anchor applies to patient? {bool(any_apply_before_union)}\n")
        f.write(f"Pre-pruning: #candidate anchors total: {len(all_medoid_anchors_for_instance)}\n")
        f.write(f"Pre-pruning: #candidate anchors applying: "
                f"{sum(anchor_applies_to_instance(a['names'], new_patient) for a in all_medoid_anchors_for_instance)}\n\n")


        # =====================================================
        # UNION MODE 1
        # =====================================================
        f.write("UNION PRUNING MODE 1 (max marginal gain)\n")

        coverage_df = subset_new_patient

        final_u1, union_cov1, cov_traj1, gain_traj1 = union_prune_anchors_mode1(
            all_medoid_anchors_for_instance,
            coverage_df,
            n_samples=union_samples,
            flatten_tol=flatten_tol
        )

        if len(final_u1) == 0:
            f.write("No anchors selected by union pruning MODE 1.\n\n")
            results.append({
                'instance_idx': new_patient_idx,
                'mode': 'union_mode1',
                'label_to_exclude': label_to_exclude,
                'main_precision': np.nan,
                'main_coverage': np.nan,
                'cumulative_coverage': float(union_cov1),
                'avg_precision_valid': np.nan,
                'avg_feats_valid': 0.0,
                'num_unique_feats_valid': 0,
                'num_valid_anchors': 0,
                'main_applies': 0,
                'num_valid_apply': 0,
                'coverage_traj': cov_traj1,
                'gain_traj': gain_traj1,
                'any_apply_before_union': int(any_apply_before_union),
            })
        else:
            main_u1 = final_u1[0]
            main_prec_u1 = float(main_u1['precision'])
            main_cov_u1 = float(main_u1['coverage'])

            avg_feats_u1 = float(np.mean([len(a['names']) for a in final_u1]))
            unique_feats_u1 = set()
            for a in final_u1:
                for cond in a['names']:
                    unique_feats_u1.add(extract_feature_name(cond))
            num_unique_feats_u1 = len(unique_feats_u1)

            main_applies_u1 = int(anchor_applies_to_instance(main_u1['names'], new_patient))
            num_apply_u1 = int(sum(anchor_applies_to_instance(a['names'], new_patient) for a in final_u1))

            avg_prec_all_u1 = float(np.mean([a['precision'] for a in final_u1])) if len(final_u1) else np.nan

            f.write(f"#selected anchors: {len(final_u1)}\n")
            f.write(f"Main union anchor: {' AND '.join(main_u1['names'])}\n")
            f.write(f"Main precision: {main_prec_u1:.3f}\n")
            f.write(f"Main coverage: {main_cov_u1:.6f}\n")
            f.write(f"Union coverage: {union_cov1:.6f}\n")
            f.write(f"Avg precision over ALL selected union anchors: {avg_prec_all_u1}\n")
            f.write(f"Does MAIN apply to patient? {bool(main_applies_u1)}\n")
            f.write(f"#selected anchors applying: {num_apply_u1} / {len(final_u1)}\n\n")

            results.append({
                'instance_idx': new_patient_idx,
                'mode': 'union_mode1',
                'label_to_exclude': label_to_exclude,
                'main_precision': main_prec_u1,
                'main_coverage': main_cov_u1,
                'cumulative_coverage': float(union_cov1),
                'avg_precision_valid': float(avg_prec_all_u1) if not np.isnan(avg_prec_all_u1) else np.nan,
                'avg_feats_valid': float(avg_feats_u1),
                'num_unique_feats_valid': int(num_unique_feats_u1),
                'num_valid_anchors': int(len(final_u1)),
                'main_applies': int(main_applies_u1),
                'num_valid_apply': int(num_apply_u1),
                'coverage_traj': cov_traj1,
                'gain_traj': gain_traj1,
                'any_apply_before_union': int(any_apply_before_union),
            })

        # =====================================================
        # UNION MODE 2
        # =====================================================
        f.write("UNION PRUNING MODE 2 (sorted by individual coverage)\n")

        final_u2, union_cov2, cov_traj2, gain_traj2 = union_prune_anchors_mode2(
            all_medoid_anchors_for_instance,
            coverage_df,
            n_samples=union_samples,
            flatten_tol=flatten_tol
        )

        if len(final_u2) == 0:
            f.write("No anchors selected by union pruning MODE 2.\n\n")
            results.append({
                'instance_idx': new_patient_idx,
                'mode': 'union_mode2',
                'label_to_exclude': label_to_exclude,
                'main_precision': np.nan,
                'main_coverage': np.nan,
                'cumulative_coverage': float(union_cov2),
                'avg_precision_valid': np.nan,
                'avg_feats_valid': 0.0,
                'num_unique_feats_valid': 0,
                'num_valid_anchors': 0,
                'main_applies': 0,
                'num_valid_apply': 0,
                'coverage_traj': cov_traj2,
                'gain_traj': gain_traj2,
                'any_apply_before_union': int(any_apply_before_union),
            })
        else:
            main_u2 = final_u2[0]
            main_prec_u2 = float(main_u2['precision'])
            main_cov_u2 = float(main_u2['coverage'])

            avg_feats_u2 = float(np.mean([len(a['names']) for a in final_u2]))
            unique_feats_u2 = set()
            for a in final_u2:
                for cond in a['names']:
                    unique_feats_u2.add(extract_feature_name(cond))
            num_unique_feats_u2 = len(unique_feats_u2)

            main_applies_u2 = int(anchor_applies_to_instance(main_u2['names'], new_patient))
            num_apply_u2 = int(sum(anchor_applies_to_instance(a['names'], new_patient) for a in final_u2))

            avg_prec_all_u2 = float(np.mean([a['precision'] for a in final_u2])) if len(final_u2) else np.nan

            f.write(f"#selected anchors: {len(final_u2)}\n")
            f.write(f"Main union anchor: {' AND '.join(main_u2['names'])}\n")
            f.write(f"Main precision: {main_prec_u2:.3f}\n")
            f.write(f"Main coverage: {main_cov_u2:.6f}\n")
            f.write(f"Union coverage: {union_cov2:.6f}\n")
            f.write(f"Avg precision over ALL selected union anchors: {avg_prec_all_u2}\n")
            f.write(f"Does MAIN apply to patient? {bool(main_applies_u2)}\n")
            f.write(f"#selected anchors applying: {num_apply_u2} / {len(final_u2)}\n\n")

            results.append({
                'instance_idx': new_patient_idx,
                'mode': 'union_mode2',
                'label_to_exclude': label_to_exclude,
                'main_precision': main_prec_u2,
                'main_coverage': main_cov_u2,
                'cumulative_coverage': float(union_cov2),
                'avg_precision_valid': float(avg_prec_all_u2) if not np.isnan(avg_prec_all_u2) else np.nan,
                'avg_feats_valid': float(avg_feats_u2),
                'num_unique_feats_valid': int(num_unique_feats_u2),
                'num_valid_anchors': int(len(final_u2)),
                'main_applies': int(main_applies_u2),
                'num_valid_apply': int(num_apply_u2),
                'coverage_traj': cov_traj2,
                'gain_traj': gain_traj2,
                'any_apply_before_union': int(any_apply_before_union),
            })

        f.write("-" * 80 + "\n\n")

    # ---------------------------------------------------------
    # GLOBAL STATISTICS
    # ---------------------------------------------------------
    f.write("\n" + "=" * 100 + "\n")
    f.write("GLOBAL STATISTICS ACROSS ALL INSTANCES (TRUE-LABEL EXCLUSION)\n")
    f.write("=" * 100 + "\n\n")
    f.write(f"#instances skipped (no neighbors excluding TRUE label): {counter_no_ps_excluding_true}\n\n")
    for mode in ['original', 'medoid', 'union_mode1', 'union_mode2']:

        mode_results = [r for r in results if r['mode'] == mode]
        if len(mode_results) == 0:
            continue

        cov_values = [r['main_coverage'] for r in mode_results]
        prec_values = [r['main_precision'] for r in mode_results]
        cum_cov_values = [r['cumulative_coverage'] for r in mode_results]
        feats_values = [r['avg_feats_valid'] for r in mode_results]
        unique_feats_values = [r['num_unique_feats_valid'] for r in mode_results]
        valid_anchors_values = [r['num_valid_anchors'] for r in mode_results]
        max_num_valid_anchors = int(np.max(valid_anchors_values)) if len(valid_anchors_values) else 0

    
        avg_prec_valid_values = [r['avg_precision_valid'] for r in mode_results]

        main_applies_values = [r['main_applies'] for r in mode_results]
        num_valid_apply_values = [r['num_valid_apply'] for r in mode_results]

        no_anchor_applies_values = [
            int((not r['main_applies']) and (r['num_valid_apply'] == 0))
            for r in mode_results
        ]

        count_no_anchor_applies = int(np.sum(no_anchor_applies_values))
        frac_no_anchor_applies = count_no_anchor_applies / len(mode_results)

        # --- NEW: indices of test instances where AT LEAST ONE anchor applies ---
        # no_anchor_applies == 0  <->  at least one anchor applies
        indices_at_least_one_applies = sorted(
            {r['instance_idx']
             for r, no_na in zip(mode_results, no_anchor_applies_values)
             if no_na == 0}
        )

        avg_cov, std_cov = np.mean(cov_values), np.std(cov_values)
        avg_prec, std_prec = np.mean(prec_values), np.std(prec_values)
        avg_cum_cov, std_cum_cov = np.mean(cum_cov_values), np.std(cum_cov_values)
        avg_feats_valid, std_feats_valid = np.mean(feats_values), np.std(feats_values)
        avg_unique_feats_valid, std_unique_feats_valid = np.mean(unique_feats_values), np.std(unique_feats_values)
        avg_num_valid_anchors, std_num_valid_anchors = np.mean(valid_anchors_values), np.std(valid_anchors_values)
        avg_prec_valid, std_prec_valid = np.mean(avg_prec_valid_values), np.std(avg_prec_valid_values)

        f.write(f"MODE: {mode}\n")
        f.write(f"  Avg main precision: {avg_prec:.4f} (std={std_prec:.4f})\n")
        f.write(f"  Avg main coverage: {avg_cov:.4f} (std={std_cov:.4f})\n")
        f.write(f"  Avg cumulative coverage: {avg_cum_cov:.4f} (std={std_cum_cov:.4f})\n")
        f.write(f"  Avg precision over VALID anchors: {avg_prec_valid:.4f} (std={std_prec_valid:.4f})\n")
        f.write(f"  Avg #features per valid anchor: {avg_feats_valid:.4f} (std={std_feats_valid:.4f})\n")
        f.write(f"  Avg #unique features per instance (valid anchors): "
                f"{avg_unique_feats_valid:.4f} (std={std_unique_feats_valid:.4f})\n")
        f.write(f"  Avg #valid anchors per instance: "
                f"{avg_num_valid_anchors:.4f} (std={std_num_valid_anchors:.4f})\n")
        f.write(f"  Max #anchors per instance: {max_num_valid_anchors}\n")

        f.write(
            f"  #instances with NO anchors (main nor any valid) applying to patient: "
            f"{count_no_anchor_applies} / {len(mode_results)}\n"
        )
        f.write(
            f"  Fraction of instances with NO anchors applying: "
            f"{frac_no_anchor_applies:.4f}\n"
        )
        f.write(
            "  Test instance indices with AT LEAST ONE anchor applying: "
            f"{indices_at_least_one_applies}\n"
        )


        # --- Pre-pruning diagnostic (only meaningful for union modes) ---
        if mode in ['union_mode1', 'union_mode2']:
            pre_no_apply = [
                int(r.get('any_apply_before_union', 0) == 0)
                for r in mode_results
            ]
            count_pre_no_apply = int(np.sum(pre_no_apply))
            frac_pre_no_apply = count_pre_no_apply / len(mode_results)

            f.write(
                f"  #instances with NO candidate anchors (before pruning) applying to patient: "
                f"{count_pre_no_apply} / {len(mode_results)}\n"
            )
            f.write(
                f"  Fraction with NO candidate anchors applying (before pruning): "
                f"{frac_pre_no_apply:.4f}\n"
            )

        f.write("\n")

anchor_pool.shutdown()

end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"Done. Full report saved to: {output_path}")
