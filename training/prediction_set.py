import numpy as np
import pandas as pd
from model_training import X_train, y_train, X_conf_pred, y_conf_pred, train_xgbclassifier, X_test, y_test, param_dict_xgb
import matplotlib.pyplot as plt

# Train XGBClassifier
xgb_cl, _, _ = train_xgbclassifier(param_dict_xgb, X_train, y_train, X_test, y_test)

# Number of samples in the calibration (conformal prediction) set
n = y_conf_pred.shape[0]

# Desired miscoverage level (α = 0.05 → 95% confidence level)
alpha = 0.01

# Get class names
class_names = xgb_cl.classes_

# Get predicted probabilities for calibration samples
cal_softmax = xgb_cl.predict_proba(X_conf_pred)     # Array of shape (n_samples, n_classes) containing probabilities associated with each class

# Extract predicted probability corresponding to the true class for each sample
true_cal_softmax = cal_softmax[np.arange(cal_softmax.shape[0]), y_conf_pred]    # softmax output of true class

# Compute conformal scores (the higher the score, the less confident the model was about the true class)
cal_scores = 1 - true_cal_softmax                                               # 1 - softmax output of true class

# Compute quantile of scores at level ceil((n+1)(1-α))/n
q_level = np.ceil((n+1)*(1-alpha))/n
qhat = np.quantile(cal_scores, q_level, method='higher')                        # Empirical quantile of calibration scores

# Compute softmax probabilities for the test set
test_softmax = xgb_cl.predict_proba(X_test) 
# Construct prediction sets for each test instance
prediction_sets = test_softmax >= (1 - qhat)                                    # Prediction set: classes with probability >= 1 - qhat


if __name__ == "__main__":
    print("Size of calibration data:", n, ", alpha:", alpha)
    print("qhat:", qhat)
    
    # Print prediction sets with labels
    for i in range(len(prediction_sets)):
        # Get the classes in this prediction set
        classes_in_set = class_names[prediction_sets[i]]
        # print(f"Sample {i}: {list(classes_in_set)}")

    print("-" * 80)

    # Count how many samples have multiple classes, exactly one class, etc.
    set_sizes = prediction_sets.sum(axis=1)             # Count True values in each row
    n_multiple = (set_sizes > 1).sum()
    n_single = (set_sizes == 1).sum()
    n_double = (set_sizes == 2).sum()
    n_triple = (set_sizes == 3).sum()
    n_quad = (set_sizes == 4).sum()
    n_more_than_4 = (set_sizes > 4).sum()
    n_more_than_5 = (set_sizes > 5).sum()

    print(f"\nSUMMARY:")
    print(f"Samples with exactly 1 class: {n_single}")
    print(f"Samples with multiple classes: {n_multiple}")
    print(f"Samples with 2 classes: {n_double}")
    print(f"Samples with 3 classes: {n_triple}")
    print(f"Samples with 4 classes: {n_quad}")
    print(f"Samples with more than 4 classes: {n_more_than_4}")
    print(f"Samples with more than 5 classes: {n_more_than_5}")
    print(f"Total samples: {len(prediction_sets)}")
    print(f"Percentage with multiple classes: {n_multiple/len(prediction_sets)*100:.1f}%")
    
    # Compute and print mean cardinality (average number of labels per prediction set)
    mean_cardinality = set_sizes.mean()
    print(f"Mean cardinality: {mean_cardinality:.3f}")
    print("-" * 80)

    # Select specific calibration and test samples to illustrate the process
    i_cal = 2
    i_test = 19

    # Calibration example
    print("\nCalibration example:", i_cal)
    print("Predicted probabilities for each class:", cal_softmax[i_cal])
    print("True class:", y_conf_pred.iloc[i_cal])
    print("Model chooses class", np.argmax(cal_softmax[i_cal]), "with probability", np.max(cal_softmax, axis=1)[i_cal])
    print("Conformal score", cal_scores[i_cal])
    print("1 - s_i =", 1 - cal_scores[i_cal])

    # Test example
    print("\nTest example:", i_test)
    print("Predicted probabilities for each class:", test_softmax[i_test])
    print("True class:", y_test.iloc[i_test])
    print("Prediction set", class_names[prediction_sets[i_test]])
    print("1 - qhat =", 1 - qhat)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

    # (1) Compute scores on holdout data: show one calibration sample's softmax and 1 - s_i line
    ax = axes[0]
    bars = ax.bar(np.arange(len(class_names)), cal_softmax[i_cal], color="lightgray", edgecolor="k", linewidth=0.5)
    bars[y_conf_pred.iloc[i_cal]].set_color("gray")                                   # highlight true class bar
    ax.axhline(1 - cal_scores[i_cal], color="seagreen", linestyle="--", linewidth=2)  # add horizontal line at 1 - s_i
    ax.set_title("(1) compute scores on holdout data")
    ax.set_xlabel("class")
    ax.set_ylabel("softmax output")
    ax.set_xticks(np.arange(len(class_names)))

    # (2) Get quantile: histogram of scores with quantile threshold qhat
    ax = axes[1]
    ax.hist(cal_scores, bins=30, color="darkseagreen", alpha=0.9, edgecolor="white")
    ax.axvline(qhat, color="lightcoral", linestyle="--", linewidth=2)
    ax.set_title("(2) get quantile")
    ax.set_xlabel("scores, {s_i}")
    ax.set_ylabel("#")

    # (3) Construct prediction set: show one test sample's softmax, highlight included classes, and 1 - qhat
    ax = axes[2]
    mask = test_softmax[i_test] >= 1 - qhat
    colors = np.where(mask, "#72d6c9", "lightgray")
    ax.bar(np.arange(len(class_names)), test_softmax[i_test], color=colors, edgecolor="k", linewidth=0.5)
    ax.axhline(1 - qhat, color="lightcoral", linestyle="--", linewidth=2)
    ax.set_title("(3) construct prediction set")
    ax.set_xlabel("class")
    ax.set_ylabel("softmax output")
    ax.set_xticks(np.arange(len(class_names)))

    plt.tight_layout()
    plt.show()