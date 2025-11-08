from anchor.anchor.anchor_tabular import * 
from model_training import X_train, y_train, X_conf_pred, y_conf_pred, X_anchors, y_anchors, X_test, y_test
import xgboost as xgb
from subset import get_knn_subsets
from prediction_set import compute_conformal_prediction_set_batch
from model_training import categories

np.random.seed(1)

# Load the trained model
xgb_cl = xgb.XGBClassifier()
xgb_cl.load_model("xgb_model.json")
# Get class names
class_names = xgb_cl.classes_

# Select a new patient from the test set
new_patient_idx = 346
new_patient = X_test.iloc[new_patient_idx]
print(f"---- NEW PATIENT {new_patient_idx} (from test set)----")
print("New patient:\n", new_patient)
new_patient_true_label = y_test.iloc[new_patient_idx]
print("True label of new patient:", new_patient_true_label, "corresponding to", categories[new_patient_true_label])
new_patient_pred = xgb_cl.predict(new_patient.values.reshape(1, -1))[0]
print("Predicted label of new patient:", new_patient_pred, "corresponding to", categories[new_patient_pred])
print("-" * 80)

# Compute k-nearest subsets for each test instance
all_subsets, y_subsets, similarities_list = get_knn_subsets(X_test, X_anchors, y_anchors, k=100)

subset_new_patient = all_subsets[new_patient_idx]
y_subset_new_patient = y_subsets[new_patient_idx]
print("Subset for new patient:\n", subset_new_patient)

print("-" * 80)

# Desired miscoverage level (α = 0.05 → 95% confidence level)
alpha = 0.01

# Compute prediction sets and quantile threshold qhat for all test
prediction_sets, qhat = compute_conformal_prediction_set_batch(
    xgb_cl,
    X_conf_pred,
    y_conf_pred,
    subset_new_patient,
    alpha=alpha
)

print("qhat:", qhat)
print("Prediction sets shape:", prediction_sets.shape)
for i in range(len(prediction_sets)):
        # Get the classes in this prediction set
        classes_in_set = class_names[prediction_sets[i]]
        #print(f"Sample {i}: {list(classes_in_set)}")
        #print(f"Sample {i}: True class: {y_subset_new_patient[i]}")

print("-" * 80)

label_to_exclude = 6
print("Class to exclude", label_to_exclude, "corresponding to", categories[label_to_exclude])

pred_sets_names = [list(class_names[mask]) for mask in prediction_sets]
# print(pred_sets_names)

pred_set_without_target = {}
for i, pred_set in enumerate(pred_sets_names):
    if label_to_exclude not in pred_set:
        pred_set_without_target[i] = pred_set

print(f"Prediction sets excluding true label {label_to_exclude}:")
print(pred_set_without_target)

idx_example_to_anchor = np.random.choice(list(pred_set_without_target.keys()))
anchor_instance = subset_new_patient.iloc[idx_example_to_anchor].to_numpy()
print(f"Use example {idx_example_to_anchor}, which is\n {anchor_instance}")
print("True label of anchor instance:", y_subset_new_patient.iloc[idx_example_to_anchor])
print("Predicted label of anchor instance:", xgb_cl.predict(anchor_instance.reshape(1, -1))[0])

# Get feature names
feature_cols = X_train.columns.tolist()
# Define purely categorical feature names if any
categorical_names = {}

# Create explainer using all but the new patient in the subset as training data
indices = [i for i in range(len(subset_new_patient)) if i != idx_example_to_anchor]


explainer = AnchorTabularExplainer(
        class_names=class_names,     
        feature_names=feature_cols,             # list of column names
        #train_data=X_conf_pred.to_numpy(),
        train_data=subset_new_patient.iloc[indices].to_numpy(),                     
        discretizer=None,
        categorical_names=categorical_names     # dict {feature_index: list_of_categories}
    )

# Explain the anchor instance
exp = explainer.explain_instance(anchor_instance, xgb_cl, mode="conformal",
                                 query_label=label_to_exclude, qhat=qhat, 
                                 threshold=0.95, delta=0.1, tau=0.15)

print('Anchor: %s' % (' AND '.join(exp.names())))
print('Precision: %.3f' % exp.precision())
print('Coverage: %.5f' % exp.coverage())



############ ISSUES ############

# 1. Number of neighbors
# 2. Coverage and Precision ex-post computing
# 3. Reward change
# Generate a bunch and check anchors make sense
