import numpy as np
import pandas as pd
import xgboost as xgb
from model_training import X_conf_pred, y_conf_pred, X_test, y_test
import matplotlib.pyplot as plt

def compute_conformal_threshold(model, X_cal, y_cal, alpha=0.05):
    """
    Compute the conformal calibration threshold q̂ using a holdout (calibration) set.

    This threshold quantifies how "uncertain" the model can be while still maintaining
    the desired confidence level (1 - alpha). It is later used to decide which classes
    to include in the prediction set.

    Args:
        model : fitted classifier
            Trained model supporting `predict_proba()`.
        X_cal : array-like of shape (n_cal, n_features)
            Calibration (holdout) set features.
        y_cal : array-like of shape (n_cal,)
            True labels for the calibration samples.
        alpha : float, default=0.05
            Target miscoverage rate (i.e., 5% means 95% coverage).

    Returns:
        qhat : float
            Empirical quantile of nonconformality scores, used as the threshold
            for constructing conformal prediction sets.
    """
    # Number of samples in the calibration (conformal prediction) set
    n = y_cal.shape[0]

    # Get predicted probabilities for calibration samples
    cal_softmax = model.predict_proba(X_cal)                    # Array of shape (n_samples, n_classes) containing probabilities associated with each class
    
    # Extract predicted probability corresponding to the true class for each sample
    true_cal_softmax = cal_softmax[np.arange(n), y_cal]         # Softmax output of true class
    
    # Compute conformal scores (the higher the score, the less confident the model was about the true class)
    cal_scores = 1 - true_cal_softmax                           # 1 - softmax output of true class

    # Compute quantile of scores at level ceil((n+1)(1-α))/n
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')    # Empirical quantile of calibration scores

    return qhat

def construct_prediction_sets(model, X_to_test, qhat):
    """Construct conformal prediction sets given a model and precomputed threshold q̂.

    Each prediction set contains all classes whose predicted probability is at least
    (1 - qhat), ensuring that the overall coverage is approximately (1 - alpha).

    Args:
        model : fitted classifier
            Trained model supporting `predict_proba()`.
        X_to_test : array-like of shape (n_test, n_features)
            Instances for which to compute prediction sets.
        qhat : float
            Calibrated conformal threshold obtained from a calibration set.

    Returns:
        prediction_sets : np.ndarray of bool, shape (n_test, n_classes)
            Boolean mask where entry (i, j) is True if class j is included
            in the prediction set for test instance i.
    """

    # Compute softmax probabilities for the set to test
    test_softmax = model.predict_proba(X_to_test) 
    
    # Construct prediction sets for each instance of the set to test
    prediction_sets = test_softmax >= (1 - qhat)                # Prediction set: classes with probability >= 1 - qhat

    return prediction_sets

def compute_conformal_prediction_set_batch(model, X_cal, y_cal, X_to_test, alpha=0.05):
    """Convenience wrapper that performs both calibration and prediction in one step.

    This function first computes qhat using a calibration set and then constructs
    conformal prediction sets for the given test data.

    Args:
        model : fitted classifier
            Trained model supporting `predict_proba()`.
        X_cal : array-like of shape (n_cal, n_features)
            Calibration (holdout) set features.
        y_cal : array-like of shape (n_cal,)
            True labels for calibration samples.
        X_to_test : array-like of shape (n_test, n_features)
            Test data for which prediction sets are computed.
        alpha : float, default=0.05
            Miscoverage level (1 - alpha is the target confidence level).

    Returns:
        prediction_sets : np.ndarray of bool, shape (n_test, n_classes)
            Boolean mask indicating class inclusion in each prediction set.
        qhat : float
            Empirical quantile threshold from calibration.
    """
    
    # Calibrate the conformal threshold qhat on the calibration data
    qhat = compute_conformal_threshold(model, X_cal, y_cal, alpha)

    # Construct prediction sets target data
    prediction_sets = construct_prediction_sets(model, X_to_test, qhat)

    return prediction_sets, qhat


if __name__ == "__main__":

    # Load the trained model
    xgb_cl = xgb.XGBClassifier()
    xgb_cl.load_model("xgb_model.json")
    # Get class names
    class_names = xgb_cl.classes_

    # Desired miscoverage level (α = 0.05 → 95% confidence level)
    alpha = 0.01

    # Compute prediction sets and quantile threshold qhat for all test
    prediction_sets, qhat = compute_conformal_prediction_set_batch(
        xgb_cl,
        X_conf_pred,
        y_conf_pred,
        X_test,
        alpha=alpha
    )

    # Summary statistics
    n = y_conf_pred.shape[0]
    print(f"Size of calibration data: {n}, alpha: {alpha}")
    print(f"q̂ (quantile threshold): {qhat:.4f}")
    print("-" * 80)

    set_sizes = prediction_sets.sum(axis=1)
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

    # Pick sample indices for illustration
    i_cal = 2
    i_test = 19

    # Compute calbration details for visualization
    cal_softmax = xgb_cl.predict_proba(X_conf_pred)
    true_cal_softmax = cal_softmax[np.arange(len(y_conf_pred)), y_conf_pred]
    cal_scores = 1 - true_cal_softmax

    print("\nCalibration example:", i_cal)
    print("Predicted probabilities for each class:", cal_softmax[i_cal])
    print("True class:", y_conf_pred.iloc[i_cal])
    print("Model chooses class", np.argmax(cal_softmax[i_cal]), "with probability", np.max(cal_softmax[i_cal]))
    print("Conformal score", cal_scores[i_cal])
    print("1 - s_i =", 1 - cal_scores[i_cal])
    
    print("\nTest example:", i_test)
    print("Predicted probabilities for each class:", xgb_cl.predict_proba(X_test)[i_test])
    print("True class:", y_test.iloc[i_test])
    print("Prediction set:", class_names[prediction_sets[i_test]])
    print("1 - q̂ =", 1 - qhat)

    # Print prediction sets with labels
    for i in range(len(prediction_sets)):
        # Get the classes in this prediction set
        classes_in_set = class_names[prediction_sets[i]]
        # print(f"Sample {i}: {list(classes_in_set)}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

    # (1) Compute scores on holdout data
    ax = axes[0]
    bars = ax.bar(np.arange(len(class_names)), cal_softmax[i_cal], color="lightgray", edgecolor="k", linewidth=0.5)
    bars[y_conf_pred.iloc[i_cal]].set_color("gray")                                     # highlight true class bar     
    ax.axhline(1 - cal_scores[i_cal], color="seagreen", linestyle="--", linewidth=2)    # add horizontal line at 1 - s_i
    ax.set_title("(1) compute scores on holdout data")
    ax.set_xlabel("class")
    ax.set_ylabel("softmax output")
    ax.set_xticks(np.arange(len(class_names)))

    # (2) Histogram of calibration scores with quantile qhat
    ax = axes[1]
    ax.hist(cal_scores, bins=30, color="darkseagreen", alpha=0.9, edgecolor="white", density=True)
    ax.axvline(qhat, color="lightcoral", linestyle="--", linewidth=2)
    ax.set_title("(2) get quantile")
    ax.set_xlabel("conformality scores")
    ax.set_ylabel("frequency")

    # (3) Construct prediction set for test example
    test_softmax = xgb_cl.predict_proba(X_test)
    ax = axes[2]
    mask = test_softmax[i_test] >= (1 - qhat)
    colors = np.where(mask, "#72d6c9", "lightgray")
    ax.bar(np.arange(len(class_names)), test_softmax[i_test], color=colors, edgecolor="k", linewidth=0.5)
    ax.axhline(1 - qhat, color="lightcoral", linestyle="--", linewidth=2)
    ax.set_title("(3) construct prediction set")
    ax.set_xlabel("class")
    ax.set_ylabel("softmax output")
    ax.set_xticks(np.arange(len(class_names)))

    plt.tight_layout()
    plt.savefig("conformal_prediction_sets_example.pdf")
    plt.show()
    
    
    ###########################################
    # SEPARATE SAVES 
    ###########################################

    # ============================================================
    # (1) Compute scores on holdout data
    # ============================================================
    fig, ax = plt.subplots(figsize=(4, 3.2))

    bars = ax.bar(
        np.arange(len(class_names)),
        cal_softmax[i_cal],
        color="lightgray",
        edgecolor="k",
        linewidth=0.5,
    )

    bars[y_conf_pred.iloc[i_cal]].set_color("gray")

    ax.axhline(
        1 - cal_scores[i_cal],
        color="seagreen",
        linestyle="--",
        linewidth=2,
    )

    # ax.set_title("(1) compute scores on holdout data")
    ax.set_xlabel("class")
    ax.set_ylabel("softmax output")
    ax.set_xticks(np.arange(len(class_names)))

    plt.tight_layout()
    plt.savefig("conformal_step_1_scores.pdf", bbox_inches="tight")
    plt.show()


    # ============================================================
    # (2) Histogram of calibration scores with quantile qhat
    #     Histogram sums to 1
    # ============================================================
    fig, ax = plt.subplots(figsize=(4, 3.2))

    weights = np.ones_like(cal_scores) / len(cal_scores)

    ax.hist(
        cal_scores,
        bins=30,
        weights=weights,
        color="darkseagreen",
        alpha=0.9,
        edgecolor="white",
    )

    ax.axvline(
        qhat,
        color="lightcoral",
        linestyle="--",
        linewidth=2,
    )

    # ax.set_title("(2) get quantile")
    ax.set_xlabel("conformality scores")
    ax.set_ylabel("relative frequency")

    plt.tight_layout()
    plt.savefig("conformal_step_2_quantile_histogram.pdf", bbox_inches="tight")
    plt.show()


    # ============================================================
    # (3) Construct prediction set for test example
    # ============================================================
    test_softmax = xgb_cl.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(4, 3.2))

    mask = test_softmax[i_test] >= (1 - qhat)
    colors = np.where(mask, "#72d6c9", "lightgray")

    ax.bar(
        np.arange(len(class_names)),
        test_softmax[i_test],
        color=colors,
        edgecolor="k",
        linewidth=0.5,
    )

    ax.axhline(
        1 - qhat,
        color="lightcoral",
        linestyle="--",
        linewidth=2,
    )

    # ax.set_title("(3) construct prediction set")
    ax.set_xlabel("class")
    ax.set_ylabel("softmax output")
    ax.set_xticks(np.arange(len(class_names)))

    plt.tight_layout()
    plt.savefig("conformal_step_3_prediction_set.pdf", bbox_inches="tight")
    plt.show()