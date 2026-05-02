from anchor.anchor.anchor_tabular import AnchorTabularExplainer
from model_training import (
    X_train, y_train, X_conf_pred, y_conf_pred,
    X_anchors, y_anchors, X_test, y_test,
    categories,
)
import xgboost as xgb
from subset import gower_similarity_mixed_raw
from prediction_set import compute_conformal_prediction_set_batch
from dataset_basic_setting import TEST_BOUNDS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
import time

np.random.seed(10)

# ---------------------------------------------------------
# 1. BASIC SETTINGS
# ---------------------------------------------------------
generic_symptoms_cols = [
    'fever_severity', 'cough_severity', 'chest_pain_severity',
    'abdominal_pain_severity', 'fatigue_level', 'nausea'
]

# Label → disease group
labels_stomach = [1, 5, 6, 7, 8]
labels_lung = [0, 2, 3, 4, 9]

def label_group(label):
    if label in labels_stomach:
        return "stomach"
    if label in labels_lung:
        return "lung"
    return "other"

# Feature → disease group
lung_test_cols = [
    'pulmonary_function',
    'chest_xray_score',
    'sputum_neutrophil_percent',
    'wbc_count',
]

stomach_test_cols = [
    'endoscopy_score',
    'h_pylori_level',
    'hemoglobin',
    'gastric_ph',
]

# Domains from binned training data (for procedural score)
FEATURE_DOMAINS = {}
for col in X_train.columns:
    vals = sorted(X_train[col].dropna().unique())
    if len(vals) > 0:
        FEATURE_DOMAINS[col] = vals

# ---------------------------------------------------------
# 2. PROCEDURAL SCORE UTILITIES
# ---------------------------------------------------------
def parse_condition(cond):
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
    fracs = []
    for cond in predicate_names:
        frac = predicate_coverage_fraction(cond, feature_domains)
        if frac is not None:
            fracs.append(frac)
    if not fracs:
        return None
    return float(np.mean(fracs))


# ---------------------------------------------------------
# 3. HELPER: ANCHOR APPLIES TO INSTANCE
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
        else:
            return False

    return True


# ---------------------------------------------------------
# 4. RAW GOWER DISTANCE (on binned features)
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


def gower_similarity_to_all(X_anchor, x_test, feature_names, symptom_cols, TEST_BOUNDS, symptom_range=(0, 10)):
    return gower_similarity_mixed_raw(
        anchor_vectors=X_anchor,
        test_vector=x_test,
        feature_names=feature_names,
        symptom_cols=symptom_cols,
        TEST_BOUNDS=TEST_BOUNDS,
        feature_range=symptom_range
    )


# ---------------------------------------------------------
# 5. ANCHOR DISEASE-GROUP CLASSIFICATION
# ---------------------------------------------------------
def anchor_disease_group(predicate_names):
    """
    'lung'    : at least one lung test, no stomach test (generic allowed)
    'stomach' : at least one stomach test, no lung test (generic allowed)
    'mixed'   : both lung and stomach tests
    'none'    : only generic or other features
    """
    has_lung = False
    has_stomach = False

    for cond in predicate_names:
        feature, _, _, _ = parse_condition(cond)
        if feature is None:
            continue
        if feature in lung_test_cols:
            has_lung = True
        if feature in stomach_test_cols:
            has_stomach = True

    if has_lung and not has_stomach:
        return "lung"
    if has_stomach and not has_lung:
        return "stomach"
    if has_lung and has_stomach:
        return "mixed"
    return "none"


# ---------------------------------------------------------
# 6. UNION-PRUNING MODES
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

    final_anchors = []
    covered_union = set()
    last_cumulative_coverage = 0.0

    coverage_trajectory = []
    gain_trajectory = []

    sorted_t = sorted(range(len(anchors)), key=lambda t: anchors[t]['cov_train'], reverse=True)

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
# 7. LOAD MODEL
# ---------------------------------------------------------
xgb_cl = xgb.XGBClassifier()
xgb_cl.load_model("xgb_model.json")
class_names = xgb_cl.classes_

# ---------------------------------------------------------
# 8. MAIN EXPERIMENT LOOP
# ---------------------------------------------------------
n_target_instances = min(100, len(X_test))
k_neighbors = 100
alpha = 0.01
feature_cols = X_train.columns.tolist()

scores_by_mode = {
    'original': [],
    'mean': [],
    'medoid': [],
    'union1': [],
    'union2': []
}

# Global list: all cross-group procedural scores from all modes
all_scores_all_modes = []

start_time = time.time()
all_test_indices = np.random.permutation(len(X_test))
count_hetero_instances = 0

for new_patient_idx in all_test_indices:
    if count_hetero_instances >= n_target_instances:
        break

    # -----------------------------
    # 8.1. Enforce missingness in generic symptoms
    # -----------------------------
    new_patient = X_test.iloc[new_patient_idx].copy()

    if new_patient[generic_symptoms_cols].isna().sum() == 0:
        missing_cols = np.random.choice(generic_symptoms_cols, size=2, replace=False)
        for col in missing_cols:
            new_patient[col] = np.nan

    new_patient_true_label = int(y_test.iloc[new_patient_idx])

    # -----------------------------
    # 8.2. k-NN neighborhood (RAW Gower on binned)
    # -----------------------------
    sims = gower_similarity_to_all(
        X_anchor=X_anchors.to_numpy(dtype=float),
        x_test=new_patient.values.astype(float),
        feature_names=feature_cols,
        symptom_cols=generic_symptoms_cols,
        TEST_BOUNDS=TEST_BOUNDS,
        symptom_range=(0, 10)
    )

    idx_neighbors = np.argsort(-sims)[:k_neighbors]
    subset_new_patient = X_anchors.iloc[idx_neighbors].reset_index(drop=True)
    neighbors_labels = y_anchors.iloc[idx_neighbors].reset_index(drop=True)

    # -----------------------------
    # 8.3. Choose random label to exclude (from all labels)
    # -----------------------------
    label_to_exclude = int(np.random.choice(class_names))
    group_label = label_group(label_to_exclude)
    if group_label == "other":
        continue

    # -----------------------------
    # 8.4. Conformal prediction sets for neighbors
    # -----------------------------
    prediction_sets, qhat = compute_conformal_prediction_set_batch(
        xgb_cl,
        X_conf_pred,
        y_conf_pred,
        subset_new_patient,
        alpha=alpha
    )

    pred_sets_names = [list(class_names[mask]) for mask in prediction_sets]
    pred_set_without_target = {}
    for i, pred_set in enumerate(pred_sets_names):
        if label_to_exclude not in pred_set:
            pred_set_without_target[i] = pred_set

    if len(pred_set_without_target) == 0:
        continue

    idxs_neighbors_excluding_target = list(pred_set_without_target.keys())
    neighbors_excluding_target = subset_new_patient.iloc[idxs_neighbors_excluding_target].reset_index(drop=True)
    neighbors_labels_excl = neighbors_labels.iloc[idxs_neighbors_excluding_target].reset_index(drop=True)

    # -----------------------------
    # 8.5. Check heterogeneous neighborhood (lung + stomach)
    # -----------------------------
    neighbor_groups = [label_group(int(l)) for l in neighbors_labels_excl]
    has_lung = any(g == "lung" for g in neighbor_groups)
    has_stomach = any(g == "stomach" for g in neighbor_groups)

    if not (has_lung and has_stomach):
        continue

    count_hetero_instances += 1
    opposite_group = "lung" if group_label == "stomach" else "stomach"

    # -----------------------------
    # 8.6. Prepare anchor candidate, means, medoids
    # -----------------------------
    # Anchor candidate row: one neighbor excluding target, with generic symptoms
    # overwritten by the current patient's generic symptoms
    idx_example_to_anchor = np.random.choice(range(len(neighbors_excluding_target)))
    anchor_row = neighbors_excluding_target.iloc[idx_example_to_anchor].copy()
    anchor_row[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
    anchor_instance = anchor_row.to_numpy()

    # Mean-instances per label (on binned space)
    unique_labels = np.unique(neighbors_labels_excl)
    mean_instances_list = []
    for lab in unique_labels:
        mask = (neighbors_labels_excl == lab)
        cluster = neighbors_excluding_target.loc[mask]
        mean_instances_list.append(cluster.mean(axis=0))
    mean_instances_binned = np.vstack(mean_instances_list)

    # Medoids on binned space with RAW Gower
    X_cluster = neighbors_excluding_target.to_numpy(dtype=float)
    feature_names_raw = neighbors_excluding_target.columns.tolist()

    D = compute_gower_distance_matrix_raw(
        X_cluster,
        feature_names=feature_names_raw,
        symptom_cols=generic_symptoms_cols,
        TEST_BOUNDS=TEST_BOUNDS,
        symptom_range=(0, 10)
    )

    # K-Medoids init from label-wise means (find closest point in each cluster)
    init_medoids = []
    cluster_indices_list = []
    for lab in unique_labels:
        mask = (neighbors_labels_excl == lab)
        cluster_indices = np.where(mask.values)[0]
        cluster_indices_list.append(cluster_indices)
    for ci, lab in zip(cluster_indices_list, unique_labels):
        # mean for this label
        cluster = X_cluster[ci]
        m = cluster.mean(axis=0)
        d_to_all_in_cluster = gower_distance_to_all_raw(
            cluster, m,
            feature_names=feature_names_raw,
            symptom_cols=generic_symptoms_cols,
            TEST_BOUNDS=TEST_BOUNDS,
            symptom_range=(0, 10)
        )
        local_idx = int(np.argmin(d_to_all_in_cluster))
        global_idx = int(ci[local_idx])
        init_medoids.append(global_idx)

    k = len(init_medoids)
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

    final_medoids_binned_df = neighbors_excluding_target.iloc[medoids_idx].copy().reset_index(drop=True)
    # overwrite generic symptoms by patient's values
    final_medoids_binned_df[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
    final_medoids_binned = final_medoids_binned_df.to_numpy()

    # -----------------------------
    # 8.7. Build explainers
    # -----------------------------
    categorical_names = {}
    indices_train_orig = [i for i in range(len(neighbors_excluding_target)) if i != idx_example_to_anchor]

    explainer_orig = AnchorTabularExplainer(
        class_names=class_names,
        feature_names=feature_cols,
        train_data=neighbors_excluding_target.iloc[indices_train_orig].to_numpy(),
        discretizer=None,
        categorical_names=categorical_names,
    )

    explainer_mean = AnchorTabularExplainer(
        class_names=class_names,
        feature_names=feature_cols,
        train_data=neighbors_excluding_target.to_numpy(),
        discretizer=None,
        categorical_names=categorical_names,
    )

    explainer_medoid = AnchorTabularExplainer(
        class_names=class_names,
        feature_names=feature_cols,
        train_data=neighbors_excluding_target.to_numpy(),
        discretizer=None,
        categorical_names=categorical_names,
    )

    # -----------------------------
    # 8.8. ORIGINAL MODE
    # -----------------------------
    exp_original, valid_anchors_original = explainer_orig.explain_instance(
        anchor_instance,
        xgb_cl,
        mode="conformal",
        query_label=label_to_exclude,
        qhat=qhat,
        threshold=0.95,
        delta=0.1,
        tau=0.15,
        beam_size=10,
    )

    # collect cross-group procedural scores for original
    for va in valid_anchors_original:
        a_group = anchor_disease_group(va['names'])
        if a_group == opposite_group:  # cross-group
            score = anchor_procedural_score(va['names'], FEATURE_DOMAINS)
            if score is not None:
                scores_by_mode['original'].append(score)
                all_scores_all_modes.append(score)

    # -----------------------------
    # 8.9. MEAN-INSTANCES MODE
    # -----------------------------
    exp_mean, valid_anchors_mean = explainer_mean.explain_instance(
        anchor_instance,
        xgb_cl,
        mode="conformal",
        query_label=label_to_exclude,
        qhat=qhat,
        threshold=0.95,
        delta=0.1,
        tau=0.15,
        beam_size=10,
        mean_instances=mean_instances_binned,
    )

    for va in valid_anchors_mean:
        a_group = anchor_disease_group(va['names'])
        if a_group == opposite_group:
            score = anchor_procedural_score(va['names'], FEATURE_DOMAINS)
            if score is not None:
                scores_by_mode['mean'].append(score)
                all_scores_all_modes.append(score)
    # -----------------------------
    # 8.10. MEDOID MODE (per-medoid anchors)
    # -----------------------------
    all_medoid_anchors_for_instance = []

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

        # also use valid_anchors_m for medoid-mode cross-group stats
        for va in valid_anchors_m:
            a_group = anchor_disease_group(va['names'])
            if a_group == opposite_group:
                score = anchor_procedural_score(va['names'], FEATURE_DOMAINS)
                if score is not None:
                    scores_by_mode['medoid'].append(score)
                    #all_scores_all_modes.append(score)

    # -----------------------------
    # 8.11. UNION MODES 1 & 2 (on medoid anchors)
    # -----------------------------
    coverage_df = neighbors_excluding_target
    flatten_tol = 1e-4

    # UNION MODE 1
    final_anchors_u1, final_union_cov_u1, _, _ = union_prune_anchors_mode1(
        all_medoid_anchors_for_instance,
        coverage_df,
        n_samples=10000,
        flatten_tol=flatten_tol
    )

    for fa in final_anchors_u1:
        a_group = anchor_disease_group(fa['names'])
        if a_group == opposite_group:
            score = anchor_procedural_score(fa['names'], FEATURE_DOMAINS)
            if score is not None:
                scores_by_mode['union1'].append(score)
                all_scores_all_modes.append(score)
    # UNION MODE 2
    final_anchors_u2, final_union_cov_u2, _, _ = union_prune_anchors_mode2(
        all_medoid_anchors_for_instance,
        coverage_df,
        n_samples=10000,
        flatten_tol=flatten_tol
    )

    for fa in final_anchors_u2:
        a_group = anchor_disease_group(fa['names'])
        if a_group == opposite_group:
            score = anchor_procedural_score(fa['names'], FEATURE_DOMAINS)
            if score is not None:
                scores_by_mode['union2'].append(score)
                all_scores_all_modes.append(score)
# ---------------------------------------------------------
# 9. PLOT HISTOGRAMS (ONE PER MODE)
# ---------------------------------------------------------
print(f"Number of heterogeneous instances used: {count_hetero_instances}")
for mode, scores in scores_by_mode.items():
    print(f"Mode {mode}: collected {len(scores)} cross-group anchors")

for mode, scores in scores_by_mode.items():
    if len(scores) == 0:
        continue
    scores_arr = np.array(scores, dtype=float)

    bins = np.linspace(0, 1, 21)
    plt.figure(figsize=(6, 4))
    plt.hist(scores_arr, bins=20, edgecolor='black')
    plt.xlabel("Procedural score (cross-group anchors)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of procedural scores – {mode} mode")
    plt.tight_layout()
    plt.savefig(f"cross_group_procedural_scores_hist_{mode}.png", dpi=200)
    plt.show()

# One global histogram with all cross-group procedural scores (all modes)
if len(all_scores_all_modes) > 0:
    all_scores_arr = np.array(all_scores_all_modes, dtype=float)
    bins = np.linspace(0, 1, 21)
    plt.figure(figsize=(6, 4))
    plt.hist(all_scores_arr, bins=20, edgecolor='black')
    plt.xlabel("Procedural score (cross-group anchors)")
    plt.ylabel("Frequency")
    plt.title("Distribution of procedural scores – all modes combined")
    plt.tight_layout()
    plt.savefig("cross_group_procedural_scores_hist_all_modes.png", dpi=200)
    plt.show()

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")