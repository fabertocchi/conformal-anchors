from model_training import X_train, y_train, X_conf_pred, y_conf_pred, X_anchors, y_anchors, X_test, y_test
from dataset_basic_setting import SYMPTOMS_NAMES
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from prediction_set import compute_conformal_threshold, construct_prediction_sets

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
    distances = np.sum(diff / R * valid_mask, axis=1) / np.sum(valid_mask, axis=1)      # True acts as 1, False as 0 → effectively counts valid features only
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
    all_subsets, y_subsets, similarities_list = [], [], []

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
        k_nearest_indices = np.argsort(-similarities)[:k]  

        # Retrieve the corresponding feature rows and labels
        subset_df = X_anchors.iloc[k_nearest_indices].reset_index(drop=True)
        y_subset_df = y_anchors.iloc[k_nearest_indices].reset_index(drop=True)

        # Store subsets and similarities for this test instance
        all_subsets.append(subset_df)                                               # Save the subset
        y_subsets.append(y_subset_df)                                               # Save corresponding labels
        similarities_list.append(similarities[k_nearest_indices])

    return all_subsets, y_subsets, similarities_list

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


if __name__ == "__main__":
    
    k = 25              # Number of nearest neighbors

    # Compute k-nearest subsets for each test instance
    all_subsets, y_subsets, similarities_list = get_knn_subsets(X_test, X_anchors, y_anchors, k=k)

    # Load trained XGBoost model
    xgb_cl = xgb.XGBClassifier()
    xgb_cl.load_model("xgb_model.json")

    # Compute model accuracy on each subset
    accuracies, y_preds_subsets = compute_subset_accuracies(xgb_cl, all_subsets, y_subsets)
    print("Average accuracy across all subsets:", np.mean(accuracies), "\n")

    # Example: inspect first test instance
    test_idx = 0
    print("Test instance:", test_idx, X_test.iloc[test_idx], "\n")
    print("\nSubset of test instance", test_idx, ":\n")
    print(all_subsets[test_idx])
    print("Similarities:\n", similarities_list[test_idx])

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