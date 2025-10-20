from model_training import X_train, y_train, X_conf_pred, y_conf_pred, X_anchors, y_anchors, X_test, y_test
from dataset_basic_setting import SYMPTOMS_NAMES
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from prediction_set import qhat

"""Version with +-1 tolerance when checking for similar instances in anchors"""

# lens = []
# for test_idx in range(len(X_test)):
#     test_instance_symptoms = X_test.iloc[test_idx][SYMPTOMS_NAMES]
#     #print("Test instance symptoms:\n", test_instance_symptoms)
#     valid_symptoms = test_instance_symptoms.index[test_instance_symptoms.notna()]

#     # Compare only on those valid symptom columns
#     # mask = (X_anchors[valid_symptoms] == test_instance_symptoms[valid_symptoms].values).all(axis=1)
#     X_valid = X_anchors[valid_symptoms]
#     diff = np.abs(X_valid - test_instance_symptoms[valid_symptoms].values)
#     mask = (diff <= 1).all(axis=1)

#     # Create subset
#     subset_df = X_anchors[mask].reset_index(drop=True)
#     lens.append(len(subset_df))
#     print(len(subset_df))

# print("Average subset size:", np.mean(lens))
# print(subset_df)
# print(len(subset_df), "instances in the subset matching the test instance symptoms.")

"""Version with k-nn to choose similar instances in anchors"""

k = 25              # Number of nearest neighbors
all_subsets = []    # Store all subsets
y_subsets = []

for test_idx in range(len(X_test)):
    test_instance_symptoms = X_test.iloc[test_idx][SYMPTOMS_NAMES]                  # Extracts row of test_idx and selects only columns in SYMPTOMS_NAMES
    # print("Test instance symptoms:\n", test_instance_symptoms)
    valid_symptoms = test_instance_symptoms.index[test_instance_symptoms.notna()]   # Keep only non-NaN symptoms
    # print("Valid symptoms:", valid_symptoms)
    
    # Get bin values for valid symptoms of test instance
    test_vector = test_instance_symptoms[valid_symptoms].values         # it is of shape (num_valid_symptoms,)
    # print("Test vector:", test_vector)
    # print("test_vector shape", test_vector.shape)                      

    # Get bin values for valid symptoms of all anchor instances
    anchor_vectors = X_anchors[valid_symptoms].values                   # it is of shape (num_anchors, num_valid_symptoms)
    # print("anchor_vectors", anchor_vectors[:5])

    # Calculate all distances at once using broadcasting (vectorized row-wise subtraction)
    # Squared differences (Euclidean distance squared)
    distances = np.sum((anchor_vectors - test_vector) ** 2, axis=1)     # axis=1 to sum across each row
    # print("difference shape", (anchor_vectors-test_vector).shape)
    # print("Distances\n", distances)
    # print("distances shape", distances.shape)                           # it is of shape (num_anchors,)

    # Alternative: Cosine similarity (vectorized) -> problem with NaN values
    # norms_anchors = np.linalg.norm(anchor_vectors, axis=1)
    # norm_test = np.linalg.norm(test_vector)
    # distances = 1 - np.dot(anchor_vectors, test_vector) / (norms_anchors * norm_test)

    # Get indices of k nearest neighbors
    k_nearest_indices = np.argsort(distances)[:k]
    # print("Indices of k nearest neighbors:", k_nearest_indices)

    # Create subset with k nearest neighbors and corresponding labels
    subset_df = X_anchors.iloc[k_nearest_indices].reset_index(drop=True)
    y_subset_df = y_anchors.iloc[k_nearest_indices].reset_index(drop=True)

    all_subsets.append(subset_df)                                           # Save the subset
    y_subsets.append(y_subset_df)                                           # Save corresponding labels

# Access any subset: all_subsets[test_idx] gives you the subset for that test instance
# Example: subset for first test instance
# print(X_test.iloc[0])
# print("\nSubset for first test instance:")
# print(all_subsets[0])

# Compute accuracy of trained XGBoost model on each subset
xgb_cl = xgb.XGBClassifier()
xgb_cl.load_model("xgb_model.json")

y_preds_subsets = []
accuracies = []

for subset_df, y_subset_df in list(zip(all_subsets, y_subsets)):
    preds = xgb_cl.predict(subset_df)
    y_preds_subsets.append(preds)
    accuracies.append(accuracy_score(y_subset_df, preds))
    # print("Accuracy of XGBoost:", accuracy_score(y_subset_df, preds))

print("Average accuracy across all subsets:", np.mean(accuracies))
print("-"*80)


# Construct prediction set for a specific test instance
test_idx = 2

# Compute softmax probabilities for the subset corresponding to the test instance
test_softmax = xgb_cl.predict_proba(all_subsets[test_idx]) 
# Construct prediction sets for instances in the subset corresponding to the test instance
prediction_sets = test_softmax >= (1 - qhat) 

print("Shape of prediction sets", prediction_sets.shape)                                    # it is of shape (num_instances_in_subset, num_classes)

# Display the prediction sets along with true labels
class_names = xgb_cl.classes_
print(class_names)
print(type(class_names))
for i in range(len(prediction_sets)):
        # Get the classes in this prediction set
        classes_in_set = class_names[prediction_sets[i]]
        print(f"Sample {i}: {list(classes_in_set)}")
        print(f"Sample {i} true class: {y_subsets[test_idx][i]}")

