# ---------------------------------------------------------
# BEAM SIZE SENSITIVITY (UNION PRUNING MODES 1 & 2, MEDOID)
# Coverage vs beam size + Runtime vs beam size
# - 50 test instances
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
from prediction_set import compute_conformal_prediction_set_batch
from data.tests_v2 import TESTS
from binning import bin_dataset
from dataset_basic_setting import TEST_BOUNDS
from sklearn_extra.cluster import KMedoids

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import matplotlib as mpl

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
np.random.seed(1)
rng = np.random.default_rng(1)

beam_sizes = [1, 3, 5, 7, 10, 12]
n_instances = 50
k_neighbors = 100
alpha = 0.01

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
class_names = xgb_cl.classes_

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


# =========================================================
# STORAGE FOR RESULTS
# =========================================================
coverage_by_beam_u1 = {b: [] for b in beam_sizes}   # union mode 1: union coverage
runtime_by_beam_u1  = {b: [] for b in beam_sizes}

coverage_by_beam_u2 = {b: [] for b in beam_sizes}   # union mode 2: union coverage
runtime_by_beam_u2  = {b: [] for b in beam_sizes}

skipped_instances = 0

# ---------------------------------------------------------
# Select instances
# ---------------------------------------------------------
n_instances = min(n_instances, len(X_test))
selected_test_indices = np.random.choice(len(X_test), size=n_instances, replace=False)

# ---------------------------------------------------------
# OUTPUT FILE
# ---------------------------------------------------------
output_path = "beam_size_sensitivity_union_modes_medoid_new.txt"

with open(output_path, "w") as f:
    f.write("BEAM SIZE SENSITIVITY – MEDOID + UNION PRUNING MODES 1 & 2\n")
    f.write("Coverage vs beam size (union coverage) + Runtime vs beam size\n")
    f.write(f"Number of test instances: {n_instances}\n")
    f.write(f"k (neighbors): {k_neighbors}\n")
    f.write(f"alpha: {alpha}\n")
    f.write(f"Beam sizes: {beam_sizes}\n")
    f.write("Label-to-exclude: ONE random label in the same disease group (≠ true label)\n")
    f.write("Anchor construction: medoid-based neighborhood + union pruning\n")
    f.write("=" * 100 + "\n\n")

    for run_id, test_idx in enumerate(selected_test_indices, 1):
        f.write("=" * 80 + "\n")
        f.write(f"Instance {run_id}/{n_instances} (test index = {test_idx})\n")

        # Target patient (no masking here)
        new_patient = X_test.iloc[test_idx].copy()
        true_label  = int(y_test.iloc[test_idx])

        # Choose label-to-exclude (random same group)
        label_to_exclude = pick_same_group_label_to_exclude_random(true_label, rng)
        if label_to_exclude is None:
            f.write("  -> No valid same-group label to exclude. Skipping instance.\n\n")
            skipped_instances += 1
            continue

        # Build 1-row DF for get_knn_subsets
        new_patient_df = new_patient.to_frame().T
        new_patient_df.index = [test_idx]

        # kNN neighborhood (binned + raw indices)
        subsets_list, y_subsets_list, similarities_list, indices_neighbors_list = get_knn_subsets(
            new_patient_df, X_anchors, y_anchors, k=k_neighbors
        )
        subset_new_patient = subsets_list[0]                 # binned neighbors (DataFrame)
        y_subset_new_patient = y_subsets_list[0]             # neighbor labels
        indices_neighbors = indices_neighbors_list[0]        # indices into *_orig pools

        subset_new_patient_raw = X_anchors_orig.iloc[indices_neighbors].reset_index(drop=True)

        # conformal prediction sets on neighbors + qhat
        prediction_sets, qhat = compute_conformal_prediction_set_batch(
            xgb_cl, X_conf_pred, y_conf_pred, subset_new_patient, alpha=alpha
        )

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

        # Explainer for medoid mode (using full neighbor subset as train_data)
        feature_cols = X_train.columns.tolist()
        explainer_medoid = AnchorTabularExplainer(
            class_names=class_names,
            feature_names=feature_cols,
            train_data=subset_new_patient.to_numpy(),
            discretizer=None,
            categorical_names={},
        )

        # coverage_df for union pruning: use neighbor subset (binned)
        coverage_df = subset_new_patient

        # -----------------------------------------------------
        # LOOP OVER BEAM SIZES – UNION MODE 1 & 2
        # -----------------------------------------------------
        flatten_tol = 1e-4

        for b in beam_sizes:
            f.write(f"  >> Beam size B = {b}\n")

            # =======================
            # UNION MODE 1
            # =======================
            t0 = time.perf_counter()

            all_medoid_anchors_u1 = []
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
                    beam_size=b,
                    predicate_mode="medoid",
                    mean_instances=None
                )

                # collect anchors: valid + main
                for va in valid_anchors_m:
                    all_medoid_anchors_u1.append({
                        'names': va['names'],
                        'medoid_id': m_id,
                        'source': 'valid_anchor',
                        'precision': va['precision'][-1],
                        'coverage': va['coverage'][-1],
                    })

                main_names_m = exp_m.names()
                already_present = any(
                    (a['names'] == main_names_m) and (a['medoid_id'] == m_id)
                    for a in all_medoid_anchors_u1
                )
                if not already_present:
                    all_medoid_anchors_u1.append({
                        'names': main_names_m,
                        'medoid_id': m_id,
                        'source': 'main_anchor',
                        'precision': exp_m.precision(),
                        'coverage': exp_m.coverage(),
                    })

            final_anchors_u1, final_union_cov1, cov_traj_u1, gain_traj_u1 = (
                union_prune_anchors_mode1(
                    all_medoid_anchors_u1,
                    coverage_df,
                    n_samples=10000,
                    flatten_tol=flatten_tol
                )
            )

            t1 = time.perf_counter()
            rt_u1 = float(t1 - t0)

            # store union coverage (not just first anchor coverage)
            coverage_by_beam_u1[b].append(float(final_union_cov1))
            runtime_by_beam_u1[b].append(rt_u1)

            f.write(f"    [union1] union_coverage={final_union_cov1:.5f} | runtime={rt_u1:.4f}s\n")

            # =======================
            # UNION MODE 2
            # =======================
            t0 = time.perf_counter()

            all_medoid_anchors_u2 = []
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
                    beam_size=b,
                    predicate_mode="medoid",
                    mean_instances=None
                )

                for va in valid_anchors_m:
                    all_medoid_anchors_u2.append({
                        'names': va['names'],
                        'medoid_id': m_id,
                        'source': 'valid_anchor',
                        'precision': va['precision'][-1],
                        'coverage': va['coverage'][-1],
                    })

                main_names_m = exp_m.names()
                already_present = any(
                    (a['names'] == main_names_m) and (a['medoid_id'] == m_id)
                    for a in all_medoid_anchors_u2
                )
                if not already_present:
                    all_medoid_anchors_u2.append({
                        'names': main_names_m,
                        'medoid_id': m_id,
                        'source': 'main_anchor',
                        'precision': exp_m.precision(),
                        'coverage': exp_m.coverage(),
                    })

            final_anchors_u2, final_union_cov2, cov_traj_u2, gain_traj_u2 = (
                union_prune_anchors_mode2(
                    all_medoid_anchors_u2,
                    coverage_df,
                    n_samples=10000,
                    flatten_tol=flatten_tol
                )
            )

            t1 = time.perf_counter()
            rt_u2 = float(t1 - t0)

            coverage_by_beam_u2[b].append(float(final_union_cov2))
            runtime_by_beam_u2[b].append(rt_u2)

            f.write(f"    [union2] union_coverage={final_union_cov2:.5f} | runtime={rt_u2:.4f}s\n")

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

print(f"Done. Full report saved to: {output_path}")

# =========================================================
# PLOTS (union coverage vs B, runtime vs B, trade-off)
# =========================================================

def plot_coverage_vs_beam(coverage_by_beam, mean_cov, mode_name, pdf_name):
    plt.figure(figsize=(7, 5))

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
            plt.plot(beam_sizes, y, alpha=0.2)

    plt.plot(beam_sizes, mean_cov, marker='o', linewidth=2, label='Mean union coverage')

    plt.xlim(1, 12)
    plt.xticks(np.arange(1, 13, 1))
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.ylim(0, 1.0)

    plt.xlabel("Beam size B")
    plt.ylabel("Union coverage")
    plt.title(f"Union coverage vs Beam size ({mode_name})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_name)
    plt.show()


def plot_runtime_vs_beam(runtime_by_beam, mean_rt, mode_name, pdf_name):
    plt.figure(figsize=(7, 5))

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
            plt.plot(beam_sizes, y, alpha=0.2)

    plt.plot(beam_sizes, mean_rt, marker='o', linewidth=2, label='Mean runtime')

    plt.xlim(1, 12)
    plt.xticks(np.arange(1, 13, 1))
    # plt.yticks(np.arange(0, 21, 1))
    plt.ylim(0, 120)

    plt.xlabel("Beam size B")
    plt.ylabel("Runtime per explanation (seconds)")
    plt.title(f"Runtime vs Beam size ({mode_name})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pdf_name)
    plt.show()


def plot_tradeoff(mean_cov, mean_rt, mode_name, pdf_name):
    plt.figure(figsize=(7, 5))

    mean_cov_arr = np.array(mean_cov, dtype=float)
    mean_rt_arr  = np.array(mean_rt, dtype=float)
    beam_arr     = np.array(beam_sizes, dtype=int)

    plt.plot(mean_rt_arr, mean_cov_arr, marker='o', linewidth=2)

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

        plt.annotate(
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

    plt.xlabel("Mean runtime per explanation (seconds)")
    plt.ylabel("Mean union coverage (sampled)")
    plt.title(f"Coverage–runtime trade-off across beam sizes ({mode_name})", pad=20)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(pdf_name)
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

# PLOTS FOR UNION MODE 1
plot_coverage_vs_beam(
    coverage_by_beam_u1,
    mean_cov_u1_arr,
    mode_name="Union pruning mode 1",
    pdf_name="coverage_vs_beamsize_union1_medoid.pdf"
)

plot_runtime_vs_beam(
    runtime_by_beam_u1,
    mean_rt_u1_arr,
    mode_name="Union pruning mode 1",
    pdf_name="runtime_vs_beamsize_union1_medoid.pdf"
)

plot_tradeoff(
    mean_cov_u1_arr,
    mean_rt_u1_arr,
    mode_name="Union pruning mode 1",
    pdf_name="tradeoff_coverage_vs_runtime_union1_medoid.pdf"
)

# PLOTS FOR UNION MODE 2
plot_coverage_vs_beam(
    coverage_by_beam_u2,
    mean_cov_u2_arr,
    mode_name="Union pruning mode 2",
    pdf_name="coverage_vs_beamsize_union2_medoid.pdf"
)

plot_runtime_vs_beam(
    runtime_by_beam_u2,
    mean_rt_u2_arr,
    mode_name="Union pruning mode 2",
    pdf_name="runtime_vs_beamsize_union2_medoid.pdf"
)

plot_tradeoff(
    mean_cov_u2_arr,
    mean_rt_u2_arr,
    mode_name="Union pruning mode 2",
    pdf_name="tradeoff_coverage_vs_runtime_union2_medoid.pdf"
)