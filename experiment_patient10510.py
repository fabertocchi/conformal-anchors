# ---------------------------------------------------------
# Run experiment on ONLY instance 10510 and label_to_exclude=1
# Two-step procedure:
#   Step 1: original neighbors -> anchors -> collect feature lists
#   Step 2: impute patient on those lists -> dynamic neighbors -> anchors again
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
from subset import get_knn_subsets, get_knn_subsets_dynamic  
from prediction_set import compute_conformal_prediction_set_batch
import matplotlib.pyplot as plt
from data.tests_v2 import TESTS
from binning import bin_dataset
from binning import TEST_BIN_LABELS
import pandas as pd
import numpy as np
import time
from dataset_basic_setting import HEALTHY_RANGES

np.random.seed(1)
rng = np.random.default_rng(1)

# Load the trained model
xgb_cl = xgb.XGBClassifier()
xgb_cl.load_model("xgb_model.json")
class_names = xgb_cl.classes_

generic_symptoms_cols = [
    'fever_severity', 'cough_severity', 'chest_pain_severity',
    'abdominal_pain_severity', 'fatigue_level', 'nausea'
]

start_time = time.time()

# -----------------------------
# (A) STEP 1: ORIGINAL NEIGHBORS
# -----------------------------
all_subsets, y_subsets, similarities_list, indices_neighbors = get_knn_subsets(
    X_test, X_anchors, y_anchors, k=100
)

# --- ONLY ONE INSTANCE / ONE LABEL ---
new_patient_idx = 10510
label_to_exclude = 1
beam_size = 15
alpha = 0.01

results = []

output_path = "anchor_experiment_two_step_15_instance10510_label1.txt"
with open(output_path, "w") as f:
    f.write("TWO-STEP EXPERIMENT ON ANCHORS (instance=10510, label_to_exclude=1)\n")
    f.write(f"Beam size: {beam_size}\n")
    f.write("=" * 100 + "\n\n")

    f.write(f"### STEP 1: ORIGINAL neighbors -> compute anchors -> collect features\n")
    f.write("-" * 80 + "\n")

    # -----------------------------
    # 1) BASIC INSTANCE INFO
    # -----------------------------
    new_patient = X_test.iloc[new_patient_idx].copy()
    new_patient_true_label = int(y_test.iloc[new_patient_idx])
    new_patient_pred = int(xgb_cl.predict(new_patient.values.reshape(1, -1))[0])

    f.write(f"Test index: {new_patient_idx}\n")
    f.write(f"True label: {new_patient_true_label} ({categories[new_patient_true_label]})\n")
    f.write(f"Predicted label: {new_patient_pred} ({categories[new_patient_pred]})\n\n")

    # Subset + raw subset for this instance (based on ORIGINAL neighbors)
    subset_new_patient = all_subsets[new_patient_idx]
    y_subset_new_patient = y_subsets[new_patient_idx]

    subset_new_patient_raw = X_anchors_orig.iloc[
        indices_neighbors[new_patient_idx]
    ].reset_index(drop=True)

    # -----------------------------
    # 2) CONFORMAL PREDICTION SETS (on subset)
    # -----------------------------
    prediction_sets, qhat = compute_conformal_prediction_set_batch(
        xgb_cl,
        X_conf_pred,
        y_conf_pred,
        subset_new_patient,
        alpha=alpha
    )

    f.write(f"qhat: {qhat:.5f}\n")
    f.write(f"Prediction sets shape: {prediction_sets.shape}\n\n")

    # Precompute prediction set names once
    pred_sets_names = [list(class_names[mask]) for mask in prediction_sets]

    # Filter prediction sets that do NOT contain label_to_exclude
    pred_set_without_target = {}
    for i, pred_set in enumerate(pred_sets_names):
        if label_to_exclude not in pred_set:
            pred_set_without_target[i] = pred_set

    if len(pred_set_without_target) == 0:
        f.write(f"No prediction sets exclude label {label_to_exclude}. STOP.\n")
        raise SystemExit

    f.write(f"{len(pred_set_without_target)} prediction sets excluding class {label_to_exclude}:\n")
    f.write(str(pred_set_without_target) + "\n\n")

    # -----------------------------
    # 3) CHOOSE ANCHOR INSTANCE WITHIN SUBSET
    # -----------------------------
    idx_example_to_anchor = 21#int(np.random.choice(list(pred_set_without_target.keys())))

    anchor_row = subset_new_patient.iloc[idx_example_to_anchor].copy()
    anchor_row[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
    anchor_instance = anchor_row.to_numpy()

    f.write(f"Anchor candidate index within subset: {idx_example_to_anchor}\n")
    f.write(f"True label of (original) anchor neighbor: {int(y_subset_new_patient.iloc[idx_example_to_anchor])}\n")
    f.write(f"Predicted label of modified anchor instance: {int(xgb_cl.predict(anchor_instance.reshape(1, -1))[0])}\n\n")

    idxs_neighbors_excluding_target = list(pred_set_without_target.keys())
    neighbors_excluding_target = subset_new_patient.iloc[idxs_neighbors_excluding_target]
    neighbors_excluding_target_raw = subset_new_patient_raw.iloc[idxs_neighbors_excluding_target]
    neighbors_labels = y_subset_new_patient.iloc[idxs_neighbors_excluding_target]

    # -----------------------------
    # 4) MEAN INSTANCES (for mean-instances mode)
    # -----------------------------
    unique_labels = np.unique(neighbors_labels)
    mean_instances_raw = []
    for lab in unique_labels:
        mask = (neighbors_labels == lab)
        cluster = neighbors_excluding_target_raw[mask]
        mean_instances_raw.append(cluster.mean(axis=0))
    mean_instances_raw = np.vstack(mean_instances_raw)

    mean_instances_df = pd.DataFrame(mean_instances_raw, columns=X_train_orig.columns)
    mean_instances_binned_df = bin_dataset(
        mean_instances_df,
        TESTS=TESTS,
        generic_symptoms_cols=generic_symptoms_cols,
        verbose=False
    )
    mean_instances_binned_df[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values

    # -----------------------------
    # 5) BUILD EXPLAINER (same as your code)
    # -----------------------------
    feature_cols = X_train.columns.tolist()
    categorical_names = {}
    indices = [i for i in range(len(subset_new_patient)) if i != idx_example_to_anchor]

    explainer = AnchorTabularExplainer(
        class_names=class_names,
        feature_names=feature_cols,
        train_data=subset_new_patient.iloc[indices].to_numpy(),
        discretizer=None,
        categorical_names=categorical_names,
    )

    # -----------------------------
    # 6) ORIGINAL MODE (STEP 1)
    # -----------------------------
    exp_original, valid_anchors_original = explainer.explain_instance(
        anchor_instance, xgb_cl, mode="conformal",
        query_label=label_to_exclude, qhat=qhat,
        threshold=0.95, delta=0.1, tau=0.15, beam_size=beam_size,
        predicate_mode="original"
    )

    f.write("ORIGINAL MODE (STEP 1)\n")
    f.write("Main anchor: %s\n" % (' AND '.join(exp_original.names())))
    f.write("Precision: %.3f\n" % exp_original.precision())
    f.write("Coverage: %.5f\n" % exp_original.coverage())
    f.write("Cumulative coverage: %.5f\n" % exp_original.cumulative_coverage())
    if len(valid_anchors_original) > 0:
        num_feats_each = [len(va['feature']) for va in valid_anchors_original]
        avg_feats = float(np.mean(num_feats_each))
        original_anchor_features = set()
        for va in valid_anchors_original:
           for feat_idx in va['feature']:
            original_anchor_features.add(feature_cols[int(feat_idx)])
        num_unique_feats = len(original_anchor_features)
        original_anchor_features = sorted(list(original_anchor_features))
    else:
        avg_feats = 0.0
        num_unique_feats = 0
    
    f.write(f"Valid anchors (original): {len(valid_anchors_original)}\n")
    f.write(f"Avg #features per valid anchor (original): {avg_feats:.3f}\n")
    f.write(f"#unique features across valid anchors (original): {num_unique_feats}\n")
    f.write(f"original_anchor_features:\n{original_anchor_features}\n\n")

    sorted_valid_anchors_original = sorted(
        valid_anchors_original,
        key=lambda va: va['coverage'][-1],
        reverse=True
    )
    for i, va in enumerate(sorted_valid_anchors_original, 1):
        names = " AND ".join(va['names'])
        prec = va['precision'][-1]
        cov = va['coverage'][-1]
        f.write(f"  {i}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}\n")

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
    })

    # -----------------------------
    # 7) MEAN-INSTANCES MODE (STEP 1)
    # -----------------------------
    exp_mean, valid_anchors_mean = explainer.explain_instance(
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

    f.write("MEAN-INSTANCES MODE (STEP 1)\n")
    f.write("Main anchor: %s\n" % (' AND '.join(exp_mean.names())))
    f.write("Precision: %.3f\n" % exp_mean.precision())
    f.write("Coverage: %.5f\n" % exp_mean.coverage())
    f.write("Cumulative coverage: %.5f\n" % exp_mean.cumulative_coverage())
    if len(valid_anchors_mean) > 0:
        num_feats_each_m = [len(va['feature']) for va in valid_anchors_mean]
        avg_feats_m = float(np.mean(num_feats_each_m))
        mean_anchor_features = set()
        for va in valid_anchors_mean:
            for feat_idx in va['feature']:
                mean_anchor_features.add(feature_cols[int(feat_idx)])
        mean_anchor_features = sorted(list(mean_anchor_features))
        num_unique_feats_m = len(mean_anchor_features)
    else:
        avg_feats_m = 0.0
        num_unique_feats_m = 0

    f.write(f"Valid anchors (mean-instances): {len(valid_anchors_mean)}\n")
    f.write(f"Avg #features per valid anchor (mean): {avg_feats_m:.3f}\n")
    f.write(f"#unique features across valid anchors (mean): {num_unique_feats_m}\n")
    f.write(f"mean_anchor_features:\n{mean_anchor_features}\n\n")

    sorted_valid_anchors_mean = sorted(
        valid_anchors_mean,
        key=lambda va: va['coverage'][-1],
        reverse=True
    )
    for i, va in enumerate(sorted_valid_anchors_mean, 1):
        names = " AND ".join(va['names'])
        prec = va['precision'][-1]
        cov = va['coverage'][-1]
        f.write(f"  {i}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}\n")

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
    })

    f.write("-" * 80 + "\n\n")

    # =========================================================
    # STEP 2: IMPUTE + DYNAMIC NEIGHBORS + RECOMPUTE ANCHORS
    #    Do it separately for original feature list and mean feature list
    # =========================================================
    f.write("\n" + "=" * 100 + "\n")
    f.write("### STEP 2A: Impute patient using original_anchor_features, dynamic neighbors, anchors again\n")
    f.write("=" * 100 + "\n")

    # -----------------------------
    # 2A-1) Impute patient on original_anchor_features
    # -----------------------------
    labels_stomach = [1, 5, 6, 7, 8]
    labels_lung = [0, 2, 3, 4, 9]

    if new_patient_true_label in labels_lung:
        patient_group = "lung"
    elif new_patient_true_label in labels_stomach:
        patient_group = "stomach"
    else:
        patient_group = "lung"  # fallback

    SYMPTOM_IMPUTE = {
        "lung": {
            "fever_severity": 5,
            "cough_severity": 7,
            "chest_pain_severity": 6,
            "abdominal_pain_severity": 1,
            "fatigue_level": 6,
            "nausea": 2,
        },
        "stomach": {
            "fever_severity": 4,
            "cough_severity": 1,
            "chest_pain_severity": 1,
            "abdominal_pain_severity": 7,
            "fatigue_level": 6,
            "nausea": 7,
        }
    }
    new_patient_imputed_orig = new_patient.copy()
    for feat in original_anchor_features:
        print("Imputing feature:", feat)
        if feat not in new_patient_imputed_orig.index:
            continue
        if pd.isna(new_patient_imputed_orig[feat]):
            # generic symptom: sample in [0..10]
            if feat in generic_symptoms_cols:
                mu = SYMPTOM_IMPUTE[patient_group].get(feat, 5)
                # small discrete noise (±1 or ±2), clipped to [0, 10]
                noise = rng.integers(-2, 3)   # {-2, -1, 0, 1, 2}
                val = int(np.clip(mu + noise, 0, 10))
                print(feat, "imputed as", val)
                new_patient_imputed_orig[feat] = float(val)
            else:
                # test feature: sample from allowed binned labels, optionally filtered by HEALTHY_RANGES
                if feat in TEST_BIN_LABELS:
                    levels = list(TEST_BIN_LABELS[feat])
                    lo, hi = HEALTHY_RANGES[feat]
                    levels2 = [v for v in levels if (v >= lo and v <= hi)]
                    if len(levels2) > 0:
                        levels = levels2
                    new_patient_imputed_orig[feat] = float(rng.choice(levels))
                else:
                    # fallback
                    raise KeyError(f"Feature {feat} not handled in imputation logic.")

    f.write("Imputed patient (orig-feature-list):\n")
    f.write(str(new_patient_imputed_orig) + "\n\n")

    # -----------------------------
    # 2A-2) Dynamic neighbors using ONLY those extra features (non-symptoms)
    # -----------------------------
    anchor_features_for_neighbors = [c for c in original_anchor_features if c not in generic_symptoms_cols]
    print("Anchor features for dynamic neighbors (orig):", anchor_features_for_neighbors)
    # run dynamic neighbor selection for a SINGLE row (avoid redoing full X_test)
    X_test_single = pd.DataFrame([new_patient_imputed_orig], columns=X_test.columns)

    all_subsets_dyn, y_subsets_dyn, similarities_dyn, indices_neighbors_dyn = get_knn_subsets_dynamic(
        X_test_single, X_anchors, y_anchors, k=100, anchor_features=anchor_features_for_neighbors
    )

    subset_new_patient_dyn = all_subsets_dyn[0]
    y_subset_new_patient_dyn = y_subsets_dyn[0]
    print(y_subset_new_patient_dyn)

    subset_new_patient_raw_dyn = X_anchors_orig.iloc[
        indices_neighbors_dyn[0]
    ].reset_index(drop=True)

    # -----------------------------
    # 2A-3) Conformal sets on new subset
    # -----------------------------
    prediction_sets_dyn, qhat_dyn = compute_conformal_prediction_set_batch(
        xgb_cl,
        X_conf_pred,
        y_conf_pred,
        subset_new_patient_dyn,
        alpha=alpha
    )
    pred_sets_names_dyn = [list(class_names[mask]) for mask in prediction_sets_dyn]

    pred_set_without_target_dyn = {}
    for i, pred_set in enumerate(pred_sets_names_dyn):
        if label_to_exclude not in pred_set:
            pred_set_without_target_dyn[i] = pred_set
    f.write(f"{len(pred_set_without_target_dyn)} prediction sets excluding class {label_to_exclude}:\n")
    f.write(str(pred_set_without_target_dyn) + "\n\n")

    if len(pred_set_without_target_dyn) == 0:
        f.write(f"[STEP 2A] No prediction sets exclude label {label_to_exclude}. Skipping.\n\n")
    else:
        idx_example_to_anchor_dyn = int(np.random.choice(list(pred_set_without_target_dyn.keys())))

        anchor_row_dyn = subset_new_patient_dyn.iloc[idx_example_to_anchor_dyn].copy()
        anchor_row_dyn[generic_symptoms_cols] = new_patient_imputed_orig[generic_symptoms_cols].values
        anchor_instance_dyn = anchor_row_dyn.to_numpy()

        idxs_neighbors_excl_dyn = list(pred_set_without_target_dyn.keys())
        neighbors_excl_raw_dyn = subset_new_patient_raw_dyn.iloc[idxs_neighbors_excl_dyn]
        neighbors_labels_dyn = y_subset_new_patient_dyn.iloc[idxs_neighbors_excl_dyn]

        unique_labels_dyn = np.unique(neighbors_labels_dyn)
        mean_instances_raw_dyn = []
        for lab in unique_labels_dyn:
            mask = (neighbors_labels_dyn == lab)
            cluster = neighbors_excl_raw_dyn[mask]
            mean_instances_raw_dyn.append(cluster.mean(axis=0))
        mean_instances_raw_dyn = np.vstack(mean_instances_raw_dyn)

        mean_instances_df_dyn = pd.DataFrame(mean_instances_raw_dyn, columns=X_train_orig.columns)
        mean_instances_binned_df_dyn = bin_dataset(
            mean_instances_df_dyn,
            TESTS=TESTS,
            generic_symptoms_cols=generic_symptoms_cols,
            verbose=False
        )
        mean_instances_binned_df_dyn[generic_symptoms_cols] = new_patient_imputed_orig[generic_symptoms_cols].values

        feature_cols = X_train.columns.tolist()
        indices_dyn = [i for i in range(len(subset_new_patient_dyn)) if i != idx_example_to_anchor_dyn]
        explainer_dyn = AnchorTabularExplainer(
            class_names=class_names,
            feature_names=feature_cols,
            train_data=subset_new_patient_dyn.iloc[indices_dyn].to_numpy(),
            discretizer=None,
            categorical_names={},
        )

        # ORIGINAL MODE (STEP 2A)
        exp_original_dyn, valid_anchors_original_dyn = explainer_dyn.explain_instance(
            anchor_instance_dyn, xgb_cl, mode="conformal",
            query_label=label_to_exclude, qhat=qhat_dyn,
            threshold=0.95, delta=0.1, tau=0.15, beam_size=beam_size,
            predicate_mode="original"
        )

        f.write("ORIGINAL MODE (STEP 2A)\n")
        f.write("Main anchor: %s\n" % (' AND '.join(exp_original_dyn.names())))
        f.write("Precision: %.3f\n" % exp_original_dyn.precision())
        f.write("Coverage: %.5f\n" % exp_original_dyn.coverage())
        f.write("Cumulative coverage: %.5f\n" % exp_original_dyn.cumulative_coverage())
        f.write(f"Valid anchors (original): {len(valid_anchors_original_dyn)}\n\n")
    # ---------------------------------------------------------
    # STEP 2B: repeat Step 2 using mean_anchor_features
    # ---------------------------------------------------------
    f.write("\n" + "=" * 100 + "\n")
    f.write("### STEP 2B: Impute patient using mean_anchor_features, dynamic neighbors, anchors again\n")
    f.write("=" * 100 + "\n")

    new_patient_imputed_mean = new_patient.copy()
    for feat in mean_anchor_features:
        if feat not in new_patient_imputed_mean.index:
            continue
        if pd.isna(new_patient_imputed_mean[feat]):
            if feat in generic_symptoms_cols:
                mu = SYMPTOM_IMPUTE[patient_group].get(feat, 5)
                # small discrete noise (±1 or ±2), clipped to [0, 10]
                noise = rng.integers(-2, 3)   # {-2, -1, 0, 1, 2}
                val = int(np.clip(mu + noise, 0, 10))

                new_patient_imputed_orig[feat] = float(val)
            else:
                if feat in TEST_BIN_LABELS:
                    levels = list(TEST_BIN_LABELS[feat])
                    if feat in HEALTHY_RANGES:
                        lo, hi = HEALTHY_RANGES[feat]
                        levels2 = [v for v in levels if (v >= lo and v <= hi)]
                        if len(levels2) > 0:
                            levels = levels2
                    new_patient_imputed_mean[feat] = float(rng.choice(levels))
                else:
                    raise KeyError(f"Feature {feat} not handled in imputation logic.")

    f.write("Imputed patient (mean-feature-list):\n")
    f.write(str(new_patient_imputed_mean) + "\n\n")

    anchor_features_for_neighbors_m = [c for c in mean_anchor_features if c not in generic_symptoms_cols]
    X_test_single_m = pd.DataFrame([new_patient_imputed_mean], columns=X_test.columns)

    all_subsets_dyn_m, y_subsets_dyn_m, similarities_dyn_m, indices_neighbors_dyn_m = get_knn_subsets_dynamic(
        X_test_single_m, X_anchors, y_anchors, k=100, anchor_features=anchor_features_for_neighbors_m
    )

    subset_new_patient_2m = all_subsets_dyn_m[0]
    y_subset_new_patient_2m = y_subsets_dyn_m[0]
    subset_new_patient_raw_2m = X_anchors_orig.iloc[indices_neighbors_dyn_m[0]].reset_index(drop=True)

    prediction_sets_2m, qhat_2m = compute_conformal_prediction_set_batch(
        xgb_cl, X_conf_pred, y_conf_pred, subset_new_patient_2m, alpha=alpha
    )
    pred_sets_names_2m = [list(class_names[mask]) for mask in prediction_sets_2m]

    pred_set_without_target_2m = {}
    for i, pred_set in enumerate(pred_sets_names_2m):
        if label_to_exclude not in pred_set:
            pred_set_without_target_2m[i] = pred_set

    if len(pred_set_without_target_2m) == 0:
        f.write(f"[STEP 2B] No prediction sets exclude label {label_to_exclude}. Skipping.\n\n")
    else:
        idx_example_to_anchor_2m = int(np.random.choice(list(pred_set_without_target_2m.keys())))

        anchor_row_2m = subset_new_patient_2m.iloc[idx_example_to_anchor_2m].copy()
        anchor_row_2m[generic_symptoms_cols] = new_patient_imputed_mean[generic_symptoms_cols].values
        anchor_instance_2m = anchor_row_2m.to_numpy()

        idxs_neighbors_excl_2m = list(pred_set_without_target_2m.keys())
        neighbors_excl_raw_2m = subset_new_patient_raw_2m.iloc[idxs_neighbors_excl_2m]
        neighbors_labels_2m = y_subset_new_patient_2m.iloc[idxs_neighbors_excl_2m]

        unique_labels_2m = np.unique(neighbors_labels_2m)
        mean_instances_raw_2m = []
        for lab in unique_labels_2m:
            mask = (neighbors_labels_2m == lab)
            cluster = neighbors_excl_raw_2m[mask]
            mean_instances_raw_2m.append(cluster.mean(axis=0))
        mean_instances_raw_2m = np.vstack(mean_instances_raw_2m)

        mean_instances_df_2m = pd.DataFrame(mean_instances_raw_2m, columns=X_train_orig.columns)
        mean_instances_binned_df_2m = bin_dataset(
            mean_instances_df_2m, TESTS=TESTS, generic_symptoms_cols=generic_symptoms_cols, verbose=False
        )
        mean_instances_binned_df_2m[generic_symptoms_cols] = new_patient_imputed_mean[generic_symptoms_cols].values

        feature_cols = X_train.columns.tolist()
        indices_2m = [i for i in range(len(subset_new_patient_2m)) if i != idx_example_to_anchor_2m]
        explainer_2m = AnchorTabularExplainer(
            class_names=class_names,
            feature_names=feature_cols,
            train_data=subset_new_patient_2m.iloc[indices_2m].to_numpy(),
            discretizer=None,
            categorical_names={},
        )

        exp_mean_2m, valid_anchors_mean_2m = explainer_2m.explain_instance(
            anchor_instance_2m, xgb_cl, mode="conformal",
            query_label=label_to_exclude, qhat=qhat_2m,
            threshold=0.95, delta=0.1, tau=0.15, beam_size=beam_size,
            predicate_mode="mean_instances",
            mean_instances=mean_instances_binned_df_2m.to_numpy()
        )

        f.write("MEAN-INSTANCES MODE (STEP 2B)\n")
        f.write("Main anchor: %s\n" % (' AND '.join(exp_mean_2m.names())))
        f.write("Precision: %.3f\n" % exp_mean_2m.precision())
        f.write("Coverage: %.5f\n" % exp_mean_2m.coverage())
        f.write("Cumulative coverage: %.5f\n" % exp_mean_2m.cumulative_coverage())
        f.write(f"Valid anchors (mean): {len(valid_anchors_mean_2m)}\n\n")

end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print(f"\nDone. Full report saved to: {output_path}")
