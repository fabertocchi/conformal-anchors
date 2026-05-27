# ---------------------------------------------------------
# COVERAGE vs BEAM SIZE (NO SYMPTOM HIDING, NO APPLICABILITY)
# Target patients sampled from the anchor set
#
# Plots one curve per mode:
#  - Original mode: cumulative / union coverage
#  - Union pruning mode 1: final union coverage
#  - Union pruning mode 2: final union coverage
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

# =========================
# Reproducibility
# =========================
GLOBAL_SEED = 1
rng = np.random.RandomState(GLOBAL_SEED)

# =========================
# Experiment settings
# =========================
beam_sizes = [1, 3, 5, 7, 10, 12, 15]
ALPHA = 0.01
DELTA = 0.01
K_NEIGHBORS = 100

N_ANCHOR_INSTANCES = 100     # average over these many patients
N_LABELS_PER_INSTANCE = 1   # sample labels-to-exclude per patient; set None for all labels

# Union pruning params
UNION_SAMPLES = 10000
UNION_FLATTEN_TOL = 1e-4

# Feature groups (needed for Gower)
generic_symptoms_cols = [
    'fever_severity', 'cough_severity', 'chest_pain_severity',
    'abdominal_pain_severity', 'fatigue_level', 'nausea'
]

labels_stomach = [1, 5, 6, 7, 8]
labels_lung = [0, 2, 3, 4, 9]

# ---------------------------------------------------------
# Helper: anchor applies (needed only for union pruning)
# ---------------------------------------------------------
def anchor_applies_to_instance(predicate_names, patient_series):
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

    return True


# ---------------------------------------------------------
# RAW Gower helpers
# ---------------------------------------------------------
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
# UNION PRUNING – MODE 1 (max marginal gain)
# ---------------------------------------------------------
def union_prune_anchors_mode1(anchors, coverage_df, n_samples=10000, flatten_tol=1e-6):
    if not anchors or coverage_df.shape[0] == 0:
        return [], 0.0

    sampled_idx = rng.choice(range(coverage_df.shape[0]), size=n_samples, replace=True)
    coverage_data = coverage_df.iloc[sampled_idx].reset_index(drop=True)
    n_cov = coverage_data.shape[0]

    cov_idx = {}
    for t, a in enumerate(anchors):
        mask = coverage_data.apply(
            lambda row: anchor_applies_to_instance(a['names'], row),
            axis=1
        ).to_numpy(dtype=bool)
        cov_idx[t] = set(np.where(mask)[0])

    selected = []
    covered_union = set()
    last_cov = 0.0
    remaining = set(range(len(anchors)))

    while remaining:
        best_t, best_gain, best_union = None, 0.0, None
        for t in remaining:
            candidate_union = covered_union | cov_idx[t]
            cov = float(len(candidate_union)) / n_cov
            gain = cov - last_cov
            if gain > best_gain:
                best_gain, best_t, best_union = gain, t, candidate_union

        if best_t is None or best_gain <= flatten_tol:
            break

        selected.append(anchors[best_t])
        covered_union = best_union
        last_cov = float(len(covered_union)) / n_cov
        remaining.remove(best_t)

    return selected, last_cov


# ---------------------------------------------------------
# UNION PRUNING – MODE 2 (sorted by individual cov)
# ---------------------------------------------------------
def union_prune_anchors_mode2(anchors, coverage_df, n_samples=10000, flatten_tol=1e-6):
    if not anchors or coverage_df.shape[0] == 0:
        return [], 0.0

    sampled_idx = rng.choice(range(coverage_df.shape[0]), size=n_samples, replace=True)
    coverage_data = coverage_df.iloc[sampled_idx].reset_index(drop=True)
    n_cov = coverage_data.shape[0]

    # precompute individual coverages + masks
    masks = []
    ind_cov = []
    for a in anchors:
        mask = coverage_data.apply(lambda row: anchor_applies_to_instance(a['names'], row), axis=1)\
                           .to_numpy(dtype=bool)
        masks.append(mask)
        ind_cov.append(float(mask.mean()))

    order = np.argsort(ind_cov)[::-1]

    selected = []
    covered_union = np.zeros(n_cov, dtype=bool)
    last_cov = 0.0

    for t in order:
        candidate_union = covered_union | masks[t]
        cov = float(candidate_union.mean())
        gain = cov - last_cov

        if len(selected) == 0:
            selected.append(anchors[t])
            covered_union = candidate_union
            last_cov = cov
        else:
            if gain <= flatten_tol:
                break
            selected.append(anchors[t])
            covered_union = candidate_union
            last_cov = cov

    return selected, last_cov


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def sample_same_group_labels(true_label, n=1):
    """
    Sample labels from the same disease group as the true label,
    excluding the true label itself.
    """
    true_label = int(true_label)

    if true_label in labels_lung:
        group = labels_lung
    elif true_label in labels_stomach:
        group = labels_stomach
    else:
        return []

    candidates = [int(l) for l in group if int(l) != true_label]

    if len(candidates) == 0:
        return []

    if n is None or n >= len(candidates):
        return candidates

    return list(rng.choice(candidates, size=n, replace=False))


# ---------------------------------------------------------
# Run one (patient, label) at one beam size
# ---------------------------------------------------------
def run_one_pair_at_beam(new_patient_idx, label_to_exclude, beam_size, xgb_cl, class_names):
    new_patient = X_anchors.iloc[new_patient_idx].copy()
    new_patient_df = new_patient.to_frame().T
    new_patient_df.index = [new_patient_idx]

    # kNN subset around THIS patient
    # subsets_list, y_subsets_list, similarities_list, indices_neighbors_list = get_knn_subsets(
    #     new_patient_df, X_anchors, y_anchors, k=K_NEIGHBORS
    # )
    # subset_new_patient = subsets_list[0]          # binned
    # y_subset_new_patient = y_subsets_list[0]
    # indices_neighbors = indices_neighbors_list[0]

    # subset_new_patient_raw = X_anchors_orig.iloc[indices_neighbors].reset_index(drop=True)

    # Remove the target patient from the anchor set before kNN search
    keep_mask = np.ones(len(X_anchors), dtype=bool)
    keep_mask[new_patient_idx] = False

    X_anchors_without_target = X_anchors.iloc[keep_mask].reset_index(drop=True)
    y_anchors_without_target = y_anchors.iloc[keep_mask].reset_index(drop=True)
    X_anchors_orig_without_target = X_anchors_orig.iloc[keep_mask].reset_index(drop=True)

    # kNN subset around THIS patient, among anchor-set patients excluding the target
    subsets_list, y_subsets_list, similarities_list, indices_neighbors_list = get_knn_subsets(
        new_patient_df,
        X_anchors_without_target,
        y_anchors_without_target,
        k=K_NEIGHBORS
    )

    subset_new_patient = subsets_list[0]          # binned
    y_subset_new_patient = y_subsets_list[0]
    indices_neighbors = indices_neighbors_list[0]

    subset_new_patient_raw = X_anchors_orig_without_target.iloc[indices_neighbors].reset_index(drop=True)

    # conformal sets on neighbors
    prediction_sets, qhat = compute_conformal_prediction_set_batch(
        xgb_cl, X_conf_pred, y_conf_pred, subset_new_patient, alpha=ALPHA
    )
    pred_sets_names = [list(class_names[mask]) for mask in prediction_sets]

    # neighbors whose pred set excludes label_to_exclude
    idxs_excl = [i for i, ps in enumerate(pred_sets_names) if label_to_exclude not in ps]
    if len(idxs_excl) == 0:
        return {"original": np.nan, "union1": np.nan, "union2": np.nan}

    neighbors_excl = subset_new_patient.iloc[idxs_excl]
    neighbors_excl_raw = subset_new_patient_raw.iloc[idxs_excl]
    neighbors_labels = y_subset_new_patient.iloc[idxs_excl]
    unique_labels = np.unique(neighbors_labels)

    # anchor_instance = random neighbor excluding target label
    idx_example_to_anchor = int(rng.choice(idxs_excl))
    anchor_row = subset_new_patient.loc[idx_example_to_anchor].copy()
    anchor_instance = anchor_row.to_numpy()

    mean_instances_raw = []
    for lab in unique_labels:
        mask_lab = (neighbors_labels == lab)
        cluster = neighbors_excl_raw.loc[mask_lab]
        mean_instances_raw.append(cluster.mean(axis=0))
    mean_instances_raw = np.vstack(mean_instances_raw)

    # k-medoids on RAW neighbors_excl_raw with raw gower
    X_cluster_raw_df = neighbors_excl_raw.reset_index(drop=True)
    X_cluster_raw = X_cluster_raw_df.to_numpy(dtype=float)
    feature_names_raw = X_cluster_raw_df.columns.tolist()

    if X_cluster_raw.shape[0] < 2 or len(unique_labels) < 1:
        return {"original": np.nan, "union1": np.nan, "union2": np.nan}

    D = compute_gower_distance_matrix_raw(
        X_cluster_raw,
        feature_names=feature_names_raw,
        symptom_cols=generic_symptoms_cols,
        TEST_BOUNDS=TEST_BOUNDS
    )

    # init medoids: nearest point to each label-mean
    cluster_indices_list = []
    for lab in unique_labels:
        mask_lab = (neighbors_labels == lab)
        cluster_indices_list.append(np.where(mask_lab.values)[0])

    init_medoids = []
    for i, m in enumerate(mean_instances_raw):
        X_cluster_label = X_cluster_raw[cluster_indices_list[i]]
        d_to_all_in_cluster = gower_distance_to_all_raw(
            X_cluster_label, m,
            feature_names=feature_names_raw,
            symptom_cols=generic_symptoms_cols,
            TEST_BOUNDS=TEST_BOUNDS
        )
        local_idx = int(np.argmin(d_to_all_in_cluster))
        global_idx = int(cluster_indices_list[i][local_idx])
        init_medoids.append(global_idx)

    k = len(init_medoids)
    if k < 1 or k > D.shape[0]:
        return {"original": np.nan, "union1": np.nan, "union2": np.nan}

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
    final_medoids_binned_df = bin_dataset(
        final_medoids_raw_df,
        TESTS=TESTS,
        generic_symptoms_cols=generic_symptoms_cols,
        verbose=False
    )
    final_medoids_binned = final_medoids_binned_df.to_numpy()

    # explainers
    feature_cols = X_train.columns.tolist()
    categorical_names = {}

    # original/mean: exclude the anchor row from training data
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

    # ORIGINAL
    exp_original, _ = explainer_orig.explain_instance(
        anchor_instance,
        xgb_cl,
        mode="conformal",
        query_label=label_to_exclude,
        qhat=qhat,
        threshold=0.95,
        delta=DELTA,
        tau=0.15,
        beam_size=beam_size,
        predicate_mode="original",
        mean_instances=None
    )
    cov_original = float(exp_original.cumulative_coverage())

    # MEDOID + collect anchors for union
    all_medoid_anchors_for_union = []

    for m_id, medoid_instance in enumerate(final_medoids_binned, 1):
        exp_m, valid_anchors_m = explainer_medoid.explain_instance(
            medoid_instance,
            xgb_cl,
            mode="conformal",
            query_label=label_to_exclude,
            qhat=qhat,
            threshold=0.95,
            delta=DELTA,
            tau=0.15,
            beam_size=beam_size,
            predicate_mode="medoid",
            mean_instances=None
        )

        # collect valid anchors
        for va in valid_anchors_m:
            all_medoid_anchors_for_union.append({
                "names": va["names"],
                "medoid_id": m_id,
                "source": "valid_anchor",
                "precision": va["precision"][-1],
                "coverage": va["coverage"][-1],
            })

        # collect main anchor
        main_names = exp_m.names()
        if not any((a["names"] == main_names and a["medoid_id"] == m_id) for a in all_medoid_anchors_for_union):
            all_medoid_anchors_for_union.append({
                "names": main_names,
                "medoid_id": m_id,
                "source": "main_anchor",
                "precision": float(exp_m.precision()),
                "coverage": float(exp_m.coverage()),
            })

    # UNION pruning coverages
    coverage_df = subset_new_patient

    _, union_cov1 = union_prune_anchors_mode1(
        all_medoid_anchors_for_union,
        coverage_df,
        n_samples=UNION_SAMPLES,
        flatten_tol=UNION_FLATTEN_TOL
    )
    _, union_cov2 = union_prune_anchors_mode2(
        all_medoid_anchors_for_union,
        coverage_df,
        n_samples=UNION_SAMPLES,
        flatten_tol=UNION_FLATTEN_TOL
    )

    return {
        "original": cov_original,
        "union1": float(union_cov1),
        "union2": float(union_cov2),
    }


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------
def run_experiment():
    xgb_cl = xgb.XGBClassifier()
    xgb_cl.load_model("xgb_model.json")
    class_names = xgb_cl.classes_

    n_instances = min(N_ANCHOR_INSTANCES, len(X_anchors))
    selected_anchor_indices = rng.choice(len(X_anchors), size=n_instances, replace=False)

    modes = ["original", "union1", "union2"]
    raw = {m: {b: [] for b in beam_sizes} for m in modes}

    start = time.time()

    for b in beam_sizes:
        print("=" * 100)
        print(f"Running beam_size = {b}")
        print("=" * 100)

        for idx in selected_anchor_indices:
            true_label = int(y_anchors.iloc[int(idx)])
            labels = sample_same_group_labels(true_label=true_label, n=N_LABELS_PER_INSTANCE)
            for lab in labels:
                out = run_one_pair_at_beam(
                    new_patient_idx=int(idx),
                    label_to_exclude=int(lab),
                    beam_size=int(b),
                    xgb_cl=xgb_cl,
                    class_names=class_names
                )
                for m in modes:
                    raw[m][b].append(out[m])

        # progress
        for m in modes:
            vals = [v for v in raw[m][b] if not np.isnan(v)]
            print(f"  {m:7s}: n={len(vals)}  mean={np.mean(vals) if len(vals) else np.nan:.4f}")

    print(f"\nTotal runtime: {(time.time()-start)/60:.2f} minutes")

    # aggregate
    curves = {m: {"mean": [], "std": []} for m in modes}
    for m in modes:
        for b in beam_sizes:
            vals = [v for v in raw[m][b] if not np.isnan(v)]
            curves[m]["mean"].append(float(np.mean(vals)) if len(vals) else np.nan)
            curves[m]["std"].append(float(np.std(vals)) if len(vals) else np.nan)

    return raw, curves


def plot_mode(title, means, stds):
    plt.figure(figsize=(7, 5))
    plt.plot(beam_sizes, means, marker="o")
    lower = np.array(means) - np.array(stds)
    upper = np.array(means) + np.array(stds)
    plt.fill_between(beam_sizes, lower, upper, alpha=0.2)
    plt.xlabel("Beam size")
    plt.ylabel("Union Coverage")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    raw, curves = run_experiment()

    plot_mode(
        "Union coverage vs Beam size — Original mode",
        curves["original"]["mean"],
        curves["original"]["std"]
    )

    plot_mode(
        "Union coverage vs Beam size — Union pruning mode 1",
        curves["union1"]["mean"],
        curves["union1"]["std"]
    )

    plot_mode(
        "Union coverage vs Beam size — Union pruning mode 2",
        curves["union2"]["mean"],
        curves["union2"]["std"]
    )