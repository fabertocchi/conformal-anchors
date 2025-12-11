from anchor.anchor.anchor_tabular import * 
from model_training import X_train, y_train, X_conf_pred, y_conf_pred, X_anchors, y_anchors, X_test, y_test, X_train_orig, y_train_orig, X_conf_pred_orig, y_conf_pred_orig, X_anchors_orig, y_anchors_orig, X_test_orig, y_test_orig
import xgboost as xgb
from subset import get_knn_subsets
from prediction_set import compute_conformal_prediction_set_batch
from model_training import categories
import matplotlib.pyplot as plt
from data.tests_v2 import TESTS
from binning import bin_dataset
import pandas as pd

np.random.seed(1)

# Load the trained model
xgb_cl = xgb.XGBClassifier()
xgb_cl.load_model("xgb_model.json")
# Get class names
class_names = xgb_cl.classes_

generic_symptoms_cols = ['fever_severity', 'cough_severity', 'chest_pain_severity', 
    'abdominal_pain_severity', 'fatigue_level', 'nausea']


# Select a new patient from the test set
new_patient_idx = 10510#9283
new_patient = X_test.iloc[new_patient_idx]
print(f"---- NEW PATIENT {new_patient_idx} (from test set)----")
print("New patient:\n", new_patient)
new_patient_true_label = y_test.iloc[new_patient_idx]
print("True label of new patient:", new_patient_true_label, "corresponding to", categories[new_patient_true_label])
new_patient_pred = xgb_cl.predict(new_patient.values.reshape(1, -1))[0]
print("Predicted label of new patient:", new_patient_pred, "corresponding to", categories[new_patient_pred])
print("-" * 80)

# Compute k-nearest subsets for each test instance
all_subsets, y_subsets, similarities_list, indices_neighbors = get_knn_subsets(X_test, X_anchors, y_anchors, k=100)
subset_new_patient = all_subsets[new_patient_idx]
y_subset_new_patient = y_subsets[new_patient_idx]

# Compute raw (non-binned) versions of the neighbors in the subset
subset_new_patient_raw = X_anchors_orig.iloc[indices_neighbors[new_patient_idx]].reset_index(drop=True)
y_subset_new_patient_raw = y_anchors_orig.iloc[indices_neighbors[new_patient_idx]].reset_index(drop=True)

print("Raw subset for new patient:\n", subset_new_patient_raw)

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

labels_stomach = [1, 5, 6, 7, 8]
labels_lung = [0, 2, 3, 4, 9]

if new_patient_true_label in labels_stomach:
    possible_labels = [i for i in labels_stomach if i != new_patient_true_label]
    label_to_exclude = np.random.choice(possible_labels)
elif new_patient_true_label in labels_lung:
    possible_labels = [i for i in labels_lung if i != new_patient_true_label]
    label_to_exclude = np.random.choice(possible_labels)
# label_to_exclude = 7 #new_patient_true_label

print("Class to exclude", label_to_exclude, "corresponding to", categories[label_to_exclude])

pred_sets_names = [list(class_names[mask]) for mask in prediction_sets]
#print(pred_sets_names[new_patient_idx])
print("Full prediction sets:", pred_sets_names)

pred_set_without_target = {}
for i, pred_set in enumerate(pred_sets_names):
    if label_to_exclude not in pred_set:
        pred_set_without_target[i] = pred_set

print(f"{len(pred_set_without_target.keys())} Prediction sets excluding true label {label_to_exclude}:")
print(pred_set_without_target)

idx_example_to_anchor = np.random.choice(list(pred_set_without_target.keys()))
anchor_instance = subset_new_patient.iloc[idx_example_to_anchor].to_numpy()

anchor_row = subset_new_patient.iloc[idx_example_to_anchor].copy()
anchor_row[generic_symptoms_cols] = new_patient[generic_symptoms_cols].values
anchor_instance = anchor_row.to_numpy()

print(f"Use example {idx_example_to_anchor}, which is\n {anchor_instance}")
print("True label of anchor instance:", y_subset_new_patient.iloc[idx_example_to_anchor])
print("Predicted label of anchor instance:", xgb_cl.predict(anchor_instance.reshape(1, -1))[0])

idxs_neighbors_excluding_target = list(pred_set_without_target.keys())
neighbors_excluding_target = subset_new_patient.iloc[idxs_neighbors_excluding_target]
neighbors_excluding_target_raw = subset_new_patient_raw.iloc[idxs_neighbors_excluding_target]
neighbors_labels = y_subset_new_patient.iloc[idxs_neighbors_excluding_target]
#print("Neighbors excluding target class:\n", list(pred_set_without_target.keys()))
#print("similarities list", similarities_list[new_patient_idx][list(pred_set_without_target.keys())].shape)
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
        categorical_names=categorical_names,     # dict {feature_index: list_of_categories}
        )

unique_labels = np.unique(neighbors_labels)
mean_instances_raw = []

for label in unique_labels:
    mask = (neighbors_labels == label)
    cluster = neighbors_excluding_target_raw[mask]
    mean_instances_raw.append(cluster.mean(axis=0))
    print("Label", label, "num instances:", len(cluster))

mean_instances_raw = np.vstack(mean_instances_raw)
print("Mean instances (raw):", mean_instances_raw)

mean_instances_df = pd.DataFrame(
    mean_instances_raw,
    columns=X_train_orig.columns   
)

mean_instances_binned_df = bin_dataset(
    mean_instances_df,
    TESTS=TESTS,
    generic_symptoms_cols=[
        'fever_severity', 'cough_severity', 'chest_pain_severity',
        'abdominal_pain_severity', 'fatigue_level', 'nausea'
    ],
    verbose=False
)

mean_instances_binned_df[generic_symptoms_cols] = (
    new_patient[generic_symptoms_cols].values
)
print("mean instances predictions",xgb_cl.predict(mean_instances_binned_df.to_numpy()))
print("Mean instances (binned, with symptoms from new patient):")
print(mean_instances_binned_df)

beam_size = 10
#np.random.seed(1)
# Explain the anchor instance using "original" predicate mode
exp_original, valid_anchors_original = explainer.explain_instance(anchor_instance, xgb_cl, mode="conformal",
                                 query_label=label_to_exclude, qhat=qhat, 
                                 threshold=0.95, delta=0.1, tau=0.15, beam_size=beam_size, predicate_mode="original")

print('Anchor: %s' % (' AND '.join(exp_original.names())))
print('Precision: %.3f' % exp_original.precision())
print('Coverage: %.5f' % exp_original.coverage())
print('Cumulative coverage: %.5f' % exp_original.cumulative_coverage())

print("\nAll valid anchors:")
sorted_valid_anchors_original = sorted(
    valid_anchors_original,
    key=lambda va: va['coverage'][-1],   # use last coverage value
    reverse=True                         # highest coverage first
)

for i, va in enumerate(sorted_valid_anchors_original, 1):
    names = " AND ".join(va['names'])
    prec = va['precision'][-1]
    cov = va['coverage'][-1]
    print(f"{i}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}")


print("-" * 80)
#np.random.seed(1)
# Explain using MEAN INSTANCES mode
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

print('Anchor: %s' % (' AND '.join(exp_mean.names())))
print('Precision: %.3f' % exp_mean.precision())
print('Coverage: %.5f' % exp_mean.coverage())
print('Cumulative coverage: %.5f' % exp_mean.cumulative_coverage())

print("\nAll valid anchors:")
sorted_valid_anchors_mean = sorted(
    valid_anchors_mean,
    key=lambda va: va['coverage'][-1],   # use last coverage value
    reverse=True                         # highest coverage first
)
for i, va in enumerate(sorted_valid_anchors_mean, 1):
    names = " AND ".join(va['names'])
    prec = va['precision'][-1]
    cov = va['coverage'][-1]
    print(f"{i}) {names}  |  precision={prec:.3f}, coverage={cov:.5f}")


print("-" * 80)
print("Comparison:")
print(f"Original mode: {len(exp_original.names())} predicates")
print('Anchor: %s' % (' AND '.join(exp_original.names())))
print('Coverage: %.5f' % exp_original.coverage())
print('Cumulative coverage: %.5f' % exp_original.cumulative_coverage())
print(f"Mean instances mode: {len(exp_mean.names())} predicates")
print('Anchor: %s' % (' AND '.join(exp_mean.names())))
print('Coverage: %.5f' % exp_mean.coverage())
print("Cumulative coverage: %.5f\n" % exp_mean.cumulative_coverage())

# # Beam sizes to test
# beam_sizes = [1, 3, 5, 7, 10, 15, 20, 25, 30]

# # Lists to store results
# coverages_original = []
# coverages_mean = []

# for b in beam_sizes:
#     print("=" * 80)
#     print(f"Running with beam_size = {b}")
    
#     # ORIGINAL mode
#     exp_orig = explainer.explain_instance(
#         anchor_instance, xgb_cl, mode="conformal",
#         query_label=label_to_exclude, qhat=qhat,
#         threshold=0.95, delta=0.1, tau=0.15, beam_size=b,
#         predicate_mode="original"
#     )
#     cov_orig = exp_orig.coverage()
#     coverages_original.append(cov_orig)
#     print('Original mode anchor: %s' % (' AND '.join(exp_orig.names())))
#     print(f"Original mode coverage (beam={b}): {cov_orig:.5f}")

#     # MEAN INSTANCES mode
#     exp_mean = explainer.explain_instance(
#         anchor_instance, xgb_cl, mode="conformal",
#         query_label=label_to_exclude, qhat=qhat,
#         threshold=0.95, delta=0.1, tau=0.15, beam_size=b,
#         predicate_mode="mean_instances",
#         mean_instances=mean_instances_binned_df.to_numpy()
#         )
#     cov_mean = exp_mean.coverage()
#     coverages_mean.append(cov_mean)
#     print('Mean-instances mode anchor: %s' % (' AND '.join(exp_mean.names())))
#     print(f"Mean-instances mode coverage (beam={b}): {cov_mean:.5f}")

# # ---- Plot results ----
# plt.figure(figsize=(7,5))
# plt.plot(beam_sizes, coverages_original, marker='o', label='Original mode')
# plt.plot(beam_sizes, coverages_mean, marker='s', label='Mean-instances mode')
# plt.xlabel('Beam size')
# plt.ylabel('Coverage')
# plt.title('Coverage vs Beam size')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.ylim(0, 1)
# plt.tight_layout()
# plt.show()

