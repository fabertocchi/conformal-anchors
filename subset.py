from model_training import X_train, y_train, X_conf_pred, y_conf_pred, X_anchors, y_anchors, X_test, y_test, X_test_orig, X_anchors_orig, y_anchors_orig
from dataset_basic_setting import SYMPTOMS_NAMES
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from prediction_set import compute_conformal_threshold, construct_prediction_sets
from binning import TEST_BIN_LABELS
from dataset_basic_setting import TEST_BOUNDS

"""Use k-NN with Gower similarity to select similar instances in anchor set for each test sample."""

def gower_similarity(anchor_vectors, test_vector, feature_range = (0, 10)):
    """
    Compute Gower similarity between each row of anchor instance and a single test instance.
    It measures how close two samples are, returning values in [0, 1], where 1 means identical and 0 means maximally different.
    
    Args:
        anchor_vectors (np.ndarray): 2D array of shape (n_anchors, n_features).
        test_vector (np.ndarray): 1D array of shape (n_features,).
        feature_range (tuple): (min, max) range of possible feature values.

    Returns:
        similarities (np.ndarray): Similarities between test_vector and each anchor (values in [0, 1]).
    """
    min_val, max_val = feature_range
    # Compute feature range width
    R = max_val - min_val

    # Compute absolute feature-wise differences using broadcasting
    # (as if you had n_anchors copies of test_vector stacked vertically and performs element-wise subtraction)
    diff = np.abs(anchor_vectors - test_vector)

    # Create mask for valid differences (not NaN) -> it ensures that only valid features contribute to the similarity
    valid_mask = ~np.isnan(diff)
    
    # Compute Gower distance per anchor instance and convert to similarity
    #distances = np.sum(diff / R * valid_mask, axis=1) / np.sum(valid_mask, axis=1)      # True acts as 1, False as 0 → effectively counts valid features only
    den = np.sum(valid_mask, axis=1)
    distances = np.where(den > 0, np.sum(diff / R * valid_mask, axis=1) / den, 1.0)
    similarities = 1 - distances
    
    return similarities

def gower_similarity_mixed(anchor_vectors,
                           test_vector,
                           feature_names,
                           BIN_LEVELS,
                           symptom_cols,
                           symptom_range=(0, 10)):
    """
    Mixed Gower similarity:
      - symptom_cols: treated as numeric with fixed range (default 0-10)
      - all other features: treated as ordinal-binned using BIN_LEVELS (distance by bin index)

    Missing rule: a feature contributes only if BOTH anchor and test are not NaN.
    """
    A = np.asarray(anchor_vectors, dtype=float)   # (n_anchors, n_features)
    t = np.asarray(test_vector, dtype=float)      # (n_features,)

    n_anchors, n_features = A.shape

    # Output-coded arrays (we will overwrite only ordinal features)
    A_code = A.copy()
    t_code = t.copy()

    # Per-feature ranges
    ranges = np.ones(n_features, dtype=float)

    sym_R = float(symptom_range[1] - symptom_range[0])

    for j, fname in enumerate(feature_names):
        if fname in symptom_cols:
            # ORIGINAL behavior for symptoms: numeric in [0,10]
            ranges[j] = sym_R #if sym_R > 0 else 1.0
            continue

        # Binned/ordinal feature
        levels = BIN_LEVELS.get(fname, None)
        if levels is None:
            raise KeyError(f"Feature '{fname}' not found in BIN_LEVELS (needed for ordinal handling).")

        # map level -> ordinal index
        # (float conversion avoids dtype mismatches like 10 vs 10.0)
        level_to_idx = {float(v): i for i, v in enumerate(levels)}
        L = len(levels)
        ranges[j] = max(L - 1, 1)

        # Convert anchor column to codes
        col = A_code[:, j]
        for i in range(n_anchors):
            v = col[i]
            if np.isnan(v):
                continue
            col[i] = level_to_idx.get(float(v), np.nan)  # unknown -> NaN
        A_code[:, j] = col

        # Convert test value to code
        if not np.isnan(t_code[j]):
            t_code[j] = level_to_idx.get(float(t_code[j]), np.nan)

    # Gower distance on mixed-coded features
    diff = np.abs(A_code - t_code)

    valid_mask = (~np.isnan(A_code)) & (~np.isnan(t_code))
    #norm_diff = (diff / ranges) * valid_mask
    norm_diff = np.where(valid_mask, diff / ranges, 0.0)


    den = np.sum(valid_mask, axis=1)
    distances = np.where(den > 0, np.sum(norm_diff, axis=1) / den, 1.0)

    return 1.0 - distances

def gower_similarity_mixed_raw(anchor_vectors, test_vector, feature_names, symptom_cols, TEST_BOUNDS, feature_range=(0, 10), debug=False):
    """
    Compute Gower similarity between each row of anchor_vectors and a single test_vector.
    Handles RAW data with per-feature ranges:
      - symptom_cols: numeric with fixed range feature_range (default 0-10)
      - test columns in TEST_BOUNDS: numeric with range (hi - lo)

    Missing rule: a feature contributes only if BOTH anchor and test are not NaN.
    """
    min_val, max_val = feature_range
    R_sym = float(max_val - min_val)

    A = np.asarray(anchor_vectors, dtype=float)   # (n_anchors, n_features)
    t = np.asarray(test_vector, dtype=float)      # (n_features,)

    # Compute absolute feature-wise differences
    diff = np.abs(A - t)

    # Valid only if BOTH not NaN (anchor and test)
    valid_mask = (~np.isnan(A)) & (~np.isnan(t))

    # Per-feature ranges (default 1.0 to avoid div-by-zero if something is missing)
    ranges = np.ones(A.shape[1], dtype=float)
    # print("Feature names:", feature_names)
    for j, fname in enumerate(feature_names):
        if fname in symptom_cols:
            ranges[j] = R_sym if R_sym > 0 else 1.0
        elif fname in TEST_BOUNDS:
            lo, hi = TEST_BOUNDS[fname]
            r = float(hi - lo)
            ranges[j] = r if r > 0 else 1.0
        else:
            # If you have other raw numeric features, add them to TEST_BOUNDS (recommended).
            # Otherwise they will be normalized by 1.0 (i.e., not scaled).
            ranges[j] = 1.0
    # print("ranges:", ranges)

    # Compute Gower distance per anchor instance and convert to similarity
    den = np.sum(valid_mask, axis=1)
    n_zero_den = np.sum(den == 0)
    if n_zero_den > 0:
        print(f"[DEBUG] den == 0 for {n_zero_den} / {len(den)} anchors")
    if debug:
        zero_den_idx = np.where(den == 0)[0]

        if len(zero_den_idx) > 0:
            print(f"\n[DEBUG] Found {len(zero_den_idx)} anchors with den == 0")

            # Print only first few to avoid flooding
            for i in zero_den_idx[:5]:
                print("\n--- Anchor index:", i)
                print("Features compared:", feature_names)
                print("Test vector:")
                print(pd.Series(test_vector, index=feature_names))

                print("Anchor vector:")
                print(pd.Series(anchor_vectors[i], index=feature_names))

                print("Valid mask:")
                print(pd.Series(valid_mask[i], index=feature_names))

    norm_diff = np.where(valid_mask, diff / ranges, 0.0)
    distances = np.where(den > 0, np.sum(norm_diff, axis=1) / den, 1.0)
    similarities = 1 - distances

    return similarities

def get_knn_subsets(X_test, X_anchors, y_anchors, k=25):
    """
    For each test instance, find the k most similar anchor instances using Gower similarity.

    Args:
        X_test (pd.DataFrame): Test set (n_test, n_features).
        X_anchors (pd.DataFrame): Anchor pool (n_anchors, n_features).
        y_anchors (pd.Series): Labels corresponding to anchor instances.
        k (int): Number of nearest neighbors to select for each test instance.

    Returns:
        all_subsets (list[pd.DataFrames]): List of DataFrames containing the k nearest anchors per test instance.
        y_subsets(list[pd.Series]): List of label Series corresponding to each subset.
        similarities_list (list[np.ndarray]): List of similarity scores for the selected k anchors.
    """
    all_subsets, y_subsets, similarities_list, k_nearest_indices_list = [], [], [], []

    for test_idx in range(len(X_test)):
        # Select the current test instance and retain only valid (non-NaN) features
        test_instance = X_test.iloc[test_idx][SYMPTOMS_NAMES]                       
        valid_symptoms = test_instance.index[test_instance.notna()]                 

        # Extract the valid features from both test and anchor sets
        test_vector = test_instance[valid_symptoms].values                          # It is of shape (num_valid_symptoms,)
        anchor_vectors = X_anchors[valid_symptoms].values                           # It is of shape (num_anchors, num_valid_symptoms)

        # Compute Gower similarities between the test instance and all anchors
        similarities = gower_similarity(anchor_vectors, test_vector, (0, 10))

        # Identify indices of the k most similar anchors (sorted descending by similarity)
        k_nearest_indices = np.argsort(-similarities, kind='mergesort')[:k]  
        k_nearest_indices_list.append(k_nearest_indices)

        # Retrieve the corresponding feature rows and labels
        subset_df = X_anchors.iloc[k_nearest_indices].reset_index(drop=True)
        y_subset_df = y_anchors.iloc[k_nearest_indices].reset_index(drop=True)

        # Store subsets and similarities for this test instance
        all_subsets.append(subset_df)                                               # Save the subset
        y_subsets.append(y_subset_df)                                               # Save corresponding labels
        similarities_list.append(similarities[k_nearest_indices])

    return all_subsets, y_subsets, similarities_list, k_nearest_indices_list

def compute_subset_accuracies(model, all_subsets, y_subsets):
    """
    Evaluate model accuracy on each subset of k nearest anchor instances.

    Args:
        model: Trained classifier implementing predict().
        all_subsets (list[pd.DataFrame]): List of anchor subsets per test instance.
        y_subsets (list[pd.Series]): True labels corresponding to each subset.

    Returns:
        accuracies (list): Model accuracy for each subset.
        y_preds_subsets (list): Model predictions for each subset.
    """
    accuracies, y_preds_subsets = [], []

    for subset_df, y_subset_df in zip(all_subsets, y_subsets):
        preds = model.predict(subset_df)
        y_preds_subsets.append(preds)
        accuracies.append(accuracy_score(y_subset_df, preds))

    return accuracies, y_preds_subsets

def get_knn_subsets_dynamic(X_test, X_anchors, y_anchors, k=25, anchor_features=None):
    if anchor_features is None:
        anchor_features = []
    cols_to_compare = list(set(SYMPTOMS_NAMES + anchor_features))
    print(cols_to_compare)
    all_subsets, y_subsets, similarities_list, k_nearest_indices_list = [], [], [], []

    for test_idx in range(len(X_test)):
        # Use the chosen feature space (not only symptoms)
        test_instance = X_test.iloc[test_idx][cols_to_compare]
        valid_feats = test_instance.index[test_instance.notna()]

        test_vector = test_instance[valid_feats].values
        anchor_vectors = X_anchors[valid_feats].values

        similarities = gower_similarity_mixed(anchor_vectors, test_vector, feature_names=valid_feats, BIN_LEVELS=TEST_BIN_LABELS, symptom_cols=SYMPTOMS_NAMES, symptom_range=(0,10))

        k_nearest_indices = np.argsort(-similarities, kind='mergesort')[:k]
        k_nearest_indices_list.append(k_nearest_indices)

        subset_df = X_anchors.iloc[k_nearest_indices].reset_index(drop=True)
        y_subset_df = y_anchors.iloc[k_nearest_indices].reset_index(drop=True)

        all_subsets.append(subset_df)
        y_subsets.append(y_subset_df)
        similarities_list.append(similarities[k_nearest_indices])

    return all_subsets, y_subsets, similarities_list, k_nearest_indices_list

def get_knn_subsets_dynamic_raw(X_test, X_anchors, y_anchors,
                                k=25, anchor_features=None,
                                symptom_cols=None,
                                TEST_BOUNDS=None,
                                symptom_range=(0, 10)):
    if anchor_features is None:
        anchor_features = []
    if symptom_cols is None:
        symptom_cols = SYMPTOMS_NAMES
    if TEST_BOUNDS is None:
        raise ValueError("TEST_BOUNDS must be provided for RAW Gower.")

    cols_to_compare = list(set(symptom_cols + anchor_features))
    print(cols_to_compare)

    all_subsets, y_subsets, similarities_list, k_nearest_indices_list = [], [], [], []

    for test_idx in range(len(X_test)):
        test_instance = X_test.iloc[test_idx][cols_to_compare]
        valid_feats = test_instance.index[test_instance.notna()]

        test_vector = test_instance[valid_feats].values
        anchor_vectors = X_anchors[valid_feats].values

        similarities = gower_similarity_mixed_raw(
            anchor_vectors=anchor_vectors,
            test_vector=test_vector,
            feature_names=list(valid_feats),
            symptom_cols=symptom_cols,
            TEST_BOUNDS=TEST_BOUNDS,
            feature_range=symptom_range,
            debug=(test_idx==4803)
        )

        k_nearest_indices = np.argsort(-similarities, kind='mergesort')[:k]
        k_nearest_indices_list.append(k_nearest_indices)

        subset_df = X_anchors.iloc[k_nearest_indices].reset_index(drop=True)
        y_subset_df = y_anchors.iloc[k_nearest_indices].reset_index(drop=True)

        all_subsets.append(subset_df)
        y_subsets.append(y_subset_df)
        similarities_list.append(similarities[k_nearest_indices])

    return all_subsets, y_subsets, similarities_list, k_nearest_indices_list


if __name__ == "__main__":
    
    k = 25              # Number of nearest neighbors

    generic_symptoms_cols = ['fever_severity', 'cough_severity', 'chest_pain_severity',
    'abdominal_pain_severity', 'fatigue_level', 'nausea']

    # Compute k-nearest subsets for each test instance
    all_subsets, y_subsets, similarities_list, k_nearest_indices = get_knn_subsets(X_test, X_anchors, y_anchors, k=k)
    all_subsets_dynamic, y_subsets_dynamic, similarities_list_dynamic, k_nearest_indices_dynamic = get_knn_subsets_dynamic(X_test, X_anchors, y_anchors, k=k, anchor_features=['hemoglobin', 'endoscopy_score'])
    all_subsets_dynamic_raw, y_subsets_dynamic_raw, similarities_list_dynamic_raw, k_nearest_indices_dynamic_raw = get_knn_subsets_dynamic_raw(X_test_orig, X_anchors_orig, y_anchors_orig, k=k, anchor_features=['hemoglobin', 'endoscopy_score'], symptom_cols=generic_symptoms_cols, TEST_BOUNDS=TEST_BOUNDS, symptom_range=(0,10))
    
    # Load trained XGBoost model
    xgb_cl = xgb.XGBClassifier()
    xgb_cl.load_model("xgb_model.json")

    # Compute model accuracy on each subset
    accuracies, y_preds_subsets = compute_subset_accuracies(xgb_cl, all_subsets, y_subsets)
    print("Average accuracy across all subsets:", np.mean(accuracies), "\n")

    # Example: inspect first test instance
    test_idx = 0
    print("Test instance:\n", test_idx, X_test.iloc[test_idx], "\n")
    print("\nSubset of test instance", test_idx, ":\n")
    print(all_subsets[test_idx].iloc[0])
    print("\nSubset dynamic of test instance", test_idx, ":\n")
    print(all_subsets_dynamic[test_idx].iloc[0])
    print("\nSubset dynamic raw of test instance", test_idx, ":\n")
    print(all_subsets_dynamic_raw[test_idx].iloc[0])
    print("Similarities:\n", similarities_list[test_idx])
    print("Similarities dynamic:\n", similarities_list_dynamic[test_idx])
    print("Similarities dynamic raw:\n", similarities_list_dynamic_raw[test_idx])

    # Construct and visualize prediction sets for first test instance
    alpha = 0.01
    # Compute global qhat once using calibration data
    qhat = compute_conformal_threshold(xgb_cl, X_conf_pred, y_conf_pred, alpha)
    print("qhat:", qhat)
    prediction_sets = construct_prediction_sets(xgb_cl, all_subsets[test_idx], qhat)
    
    # Display classes included in each prediction set vs. true labels
    class_names = xgb_cl.classes_
    for i in range(len(prediction_sets)):
        classes_in_set = class_names[prediction_sets[i]]
        print(f"Sample {i}: Prediction set: {list(classes_in_set)}")
        print(f"Sample {i}: True class: {y_subsets[test_idx][i]}")

    test_idx = 0
    cols_to_compare = list(set(generic_symptoms_cols + ['hemoglobin', 'endoscopy_score']))
    test_instance_raw = X_test_orig.iloc[test_idx][cols_to_compare]
    valid_feats_raw = test_instance_raw.index[test_instance_raw.notna()]

    test_instance_bin = X_test.iloc[test_idx][cols_to_compare]
    valid_feats_bin = test_instance_bin.index[test_instance_bin.notna()]

    test_vector_raw = test_instance_raw[valid_feats_raw].values
    anchor_vectors_raw = X_anchors_orig[valid_feats_raw].values

    test_vector_bin = test_instance_bin[valid_feats_bin].values
    anchor_vectors_bin = X_anchors[valid_feats_bin].values

    # NEW
    sim_new = gower_similarity_mixed_raw(
        anchor_vectors=anchor_vectors_raw,
        test_vector=test_vector_raw,
        feature_names=list(valid_feats_raw),
        symptom_cols=generic_symptoms_cols,
        TEST_BOUNDS=TEST_BOUNDS,
        feature_range=(0, 10)
    )

    # OLD (if you keep it)
    sim_old = gower_similarity_mixed(
        anchor_vectors=anchor_vectors_bin,
        test_vector=test_vector_bin,
        feature_names=list(valid_feats_bin),
        BIN_LEVELS=TEST_BIN_LABELS,
        symptom_cols=generic_symptoms_cols,
        symptom_range=(0, 10)
    )
    print("Top-10 indices (binned):", np.argsort(-sim_old)[:10])
    print("Top-10 similarities (binned):", np.sort(sim_old)[-10:][::-1])
    print("Top-10 indices (raw):", np.argsort(-sim_new)[:10])
    print("Top-10 similarities (raw):", np.sort(sim_new)[-10:][::-1])
    
    print("Best neighbor according to raw similarity:", k_nearest_indices_dynamic_raw[test_idx][0], "with similarity", similarities_list_dynamic_raw[test_idx][0])
    print("Test instance raw:\n", X_test_orig.iloc[test_idx])
    print("Best neighbor raw:\n", X_anchors_orig.iloc[k_nearest_indices_dynamic_raw[test_idx][0]])