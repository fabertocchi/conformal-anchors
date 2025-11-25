import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Upload the two versions of the dataset
# df_v1 = pd.read_csv("generated_datasets/synthetic_patient_basic_setting_v1.csv")
df_v2_orig = pd.read_csv("generated_datasets/synthetic_patient_basic_setting_v2.csv")

# Upload the binned version of the dataset
df_v2 = pd.read_csv("generated_datasets/synthetic_patient_basic_setting_v2_binned.csv")

# Define X and y for binned dataset
y = df_v2['disease']
X = df_v2.drop(columns=['disease', 'patient_id', 'disease_group'])
classes = y.unique()

# Define X and y for original dataset
y_orig = df_v2_orig['disease']
X_orig = df_v2_orig.drop(columns=['disease', 'patient_id', 'disease_group'])

# Convert categorical target variable to numerical: it is needed for XGBoost
y_cat = y.astype('category')       # Data type of elements of y becomes 'category' (order is defined by order of categories)
codes = y_cat.cat.codes            # Each integer code represents the category of the corresponding element observation
categories = y_cat.cat.categories  # Index of category labels

if y.dtype == 'O':
    y = y.astype('category').cat.codes  # maps categories to 0,...,n-1

if y_orig.dtype == 'O':
    y_orig = y_orig.astype('category').cat.codes  # maps categories to 0,...,n-1

# Split the binned data: we want training 65%, testing 15%, pool for anchors 10%, set for conformal predictions 10%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.65, random_state=42, stratify=y)
X_test, X_temp2, y_test, y_temp2 = train_test_split(X_temp, y_temp, train_size=0.4286, random_state=42, stratify=y_temp)
X_anchors, X_conf_pred, y_anchors, y_conf_pred = train_test_split(X_temp2, y_temp2, train_size=0.5, random_state=42, stratify=y_temp2)

# Split the original data: we want training 65%, testing 15%, pool for anchors 10%, set for conformal predictions 10%
X_train_orig, X_temp_orig, y_train_orig, y_temp_orig = train_test_split(X_orig, y_orig, train_size=0.65, random_state=42, stratify=y_orig)
X_test_orig, X_temp2_orig, y_test_orig, y_temp2_orig = train_test_split(X_temp_orig, y_temp_orig, train_size=0.4286, random_state=42, stratify=y_temp_orig)
X_anchors_orig, X_conf_pred_orig, y_anchors_orig, y_conf_pred_orig = train_test_split(X_temp2_orig, y_temp2_orig, train_size=0.5, random_state=42, stratify=y_temp2_orig)

# Define parameters for XGBClassifier model
param_dict_xgb = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y)),
    'eval_metric': 'mlogloss',
    'n_estimators': 200,
    'min_child_weight': 6,
    'reg_alpha': 0.3,
    'reg_lambda': 0.1,
    'learning_rate': 0.03,
    'max_depth': 5,
    'tree_method': 'hist',
    'random_state': 42,
    'subsample': 0.7,
    'colsample_bytree': 0.8,
    'gamma': 0.2
}

# Define parameters for RandomForestClassifier model
param_dict_rf = {
    'n_estimators': 100,
    'min_samples_leaf': 5,
    'max_depth': 10,
    'random_state': 42,
    'n_jobs': -1
}

def train_xgbclassifier(param_dict, X_train, y_train, X_test, y_test):
    """
    Train an XGBoost classifier using the parameters provided in param_dict and evaluate it on a test set.

    Parameters
    ----------
    param_dict : dict
        Dictionary containing model hyperparameters. Expected keys include:
        - 'objective' : learning objective (e.g., 'multi:softprob')
        - 'num_class' : number of target classes (for multiclass problems)
        - 'eval_metric' : evaluation metric (e.g., 'mlogloss', 'merror')
        - 'n_estimators' : number of boosting rounds (trees)
        - 'min_child_weight' : minimum sum of instance weight (hessian) needed in a child
        - 'reg_alpha' : L1 regularization term on weights
        - 'reg_lambda' : L2 regularization term on weights
        - 'learning_rate' : step size shrinkage used to prevent overfitting
        - 'max_depth' : maximum depth of individual trees
        - 'tree_method' : tree construction algorithm (e.g., 'hist', 'gpu_hist')
        - 'random_state' : random seed for reproducibility
    X_train : array-like of shape (n_samples_train, n_features)
        Training feature matrix.
    y_train : array-like of shape (n_samples_train,)
        Training target vector.
    X_test : array-like of shape (n_samples_test, n_features)
        Test feature matrix.
    y_test : array-like of shape (n_samples_test,)
        Test target vector.

    Returns
    -------
    xgb_cl : xgb.XGBClassifier
        Trained XGBoost classifier.
    proba_xgb : ndarray of shape (n_samples_test, n_classes)
        Predicted class probabilities for the test set.
    preds_xgb : ndarray of shape (n_samples_test,)
        Predicted class labels for the test set (chosen as the class with highest probability).
    """

    # Initialize the model with provided hyperparameters
    xgb_cl = xgb.XGBClassifier(
        objective=param_dict['objective'],
        num_class=param_dict['num_class'],
        eval_metric=param_dict['eval_metric'],              # Primary optimization metric
        n_estimators=param_dict['n_estimators'],
        min_child_weight=param_dict['min_child_weight'],
        reg_alpha=param_dict['reg_alpha'],                  # L1 regularization
        reg_lambda=param_dict['reg_lambda'],                # L2 regularization
        learning_rate=param_dict['learning_rate'],
        max_depth=param_dict['max_depth'],
        tree_method=param_dict['tree_method'],
        random_state=param_dict['random_state'],
        subsample=param_dict['subsample'],
        colsample_bytree=param_dict['colsample_bytree'],
        gamma=param_dict['gamma']                           # Minimum loss reduction to make a further partition
    )

    # Fit the model to the training data, monitoring performance on the test set 
    xgb_cl.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],                       # Validation data for evaluation metric
        verbose=False,
    )

    # Predict class probabilities for the test set
    proba_xgb = xgb_cl.predict_proba(X_test)  

    # Select the most probable class as the predicted label for each sample
    preds_xgb = proba_xgb.argmax(axis=1)

    # Return the trained model, predicted probabilities, and predicted labels
    return xgb_cl, proba_xgb, preds_xgb

def train_RandomForest(param_dict, X_train, y_train, X_test):
    """
    Train a Random Forest classifier using the parameters provided in param_dict and generate predictions on a test set.

    Parameters
    ----------
    param_dict : dict
        Dictionary containing model hyperparameters. Expected keys include:
        - 'n_estimators' : int
            Number of trees in the forest.
        - 'random_state' : int
            Seed for reproducibility.
        - 'min_samples_leaf' : int or float
            Minimum number (or fraction) of samples required to be at a leaf node.
        - 'max_depth' : int or None
            Maximum depth of each decision tree. Controls model complexity.
        - 'n_jobs' : int
            Number of CPU cores used for training (-1 uses all available cores).
    X_train : array-like of shape (n_samples_train, n_features)
        Training feature matrix.
    y_train : array-like of shape (n_samples_train,)
        Training target vector.
    X_test : array-like of shape (n_samples_test, n_features)
        Test feature matrix.

    Returns
    -------
    rf : RandomForestClassifier
        Trained Random Forest classifier.
    y_pred_rf : ndarray of shape (n_samples_test,)
        Predicted class labels for the test set.
    """

    # Initialize the model with the given hyperparameters
    rf = RandomForestClassifier(
        n_estimators=param_dict['n_estimators'],        # number of trees
        random_state=param_dict['random_state'],
        min_samples_leaf=param_dict['min_samples_leaf'],
        max_depth=param_dict['max_depth'],
        n_jobs=param_dict['n_jobs']                     # use all CPU cores
    )

    # Fit the model to the training data
    rf.fit(X_train, y_train)

    # Predict class labels for the test set
    y_pred_rf = rf.predict(X_test)

    # Return the trained model and predictions
    return rf, y_pred_rf

def train_model(model_name, param_dict, X_train, y_train, X_test, y_test):
    """
    Train a machine learning model based on the specified model name and parameters.

    Parameters
    ----------
    model_name : str
        Name of the model to train. Supported options:
        - 'xgboost' : trains an XGBoost classifier
        - 'random_forest' : trains a Random Forest classifier
    param_dict : dict
        Dictionary of hyperparameters specific to the chosen model.
    X_train : array-like of shape (n_samples_train, n_features)
        Training feature matrix.
    y_train : array-like of shape (n_samples_train,)
        Training target vector.
    X_test : array-like of shape (n_samples_test, n_features)
        Test feature matrix.
    y_test : array-like of shape (n_samples_test,)
        Test target vector (used only for XGBoost evaluation).

    Returns
    -------
    Depending on the model:
    - XGBoost: (trained_model, predicted_probabilities, predicted_labels)
    - Random Forest: (trained_model, predicted_labels)
    """

    if model_name == 'xgboost':
        return train_xgbclassifier(param_dict, X_train, y_train, X_test, y_test)
    elif model_name == 'random_forest':
        return train_RandomForest(param_dict, X_train, y_train, X_test)
    else:
        # Raise an error if the model name is not supported
        raise ValueError("Unsupported model name. Use 'xgboost' or 'random_forest'.")


if __name__ == '__main__':

    """About the dataset:"""
    # To display all variables
    print(f"\nVariables: {list(X.columns)}")
    # To display the values the target variable can take
    print("\nDifferent classes for target y:", classes, "\n")
    # To check which category is mapped to which code
    for code, category in enumerate(categories):
        print("Category:", category, "-> Code:", code)

    """About the splitting of the dataset:"""
    print(f"\nTraining set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Testing set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"Anchors set: {len(X_anchors)} samples ({len(X_anchors)/len(X)*100:.1f}%)")
    print(f"Conformal predictions set: {len(X_conf_pred)} samples ({len(X_conf_pred)/len(X)*100:.1f}%)")
    print(f"\nTotal: {len(X_train) + len(X_test) + len(X_anchors) + len(X_conf_pred)} samples")

    """Random Forest Classifier training and evaluation"""

    # Train the model
    rf, y_pred_rf = train_RandomForest(param_dict_rf, X_train, y_train, X_test)

    # Evaluate the model
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("Accuracy of Random Forest:", acc_rf)
    print("\nClassification report of Random forest:")
    print(classification_report(y_test, y_pred_rf))
    
    # Compute and plot confusion matrix
    cm_rf = confusion_matrix(y_test,y_pred_rf)
    ConfusionMatrixDisplay(cm_rf).plot()
    plt.title("Confusion Matrix - Random Forest")
    plt.show()

    # To check the model does not overfit
    # Predict on the training set
    y_train_pred_rf = rf.predict(X_train)
    acc_rf_train = accuracy_score(y_train, y_train_pred_rf)
    print("Random Forest Training Accuracy:", acc_rf_train)
    print("\nRandom Forest Training Classification Report:")
    print(classification_report(y_train, y_train_pred_rf))

    # Compute learning curve
    train_sizes, train_scores, val_scores = learning_curve(rf, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

    plt.plot(train_sizes, train_scores.mean(axis=1), label="Training")
    plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - Random Forest")
    plt.legend()
    plt.grid()
    plt.show()

    """XGBoost Classifier training and evaluation"""
    
    # Train the model
    xgb_cl, proba_xgb, preds_xgb = train_xgbclassifier(param_dict_xgb, X_train, y_train, X_test, y_test)

    # Save the model
    xgb_cl.save_model("xgb_model.json")

    # Evaluate the model
    print("Accuracy of XGBoost:", accuracy_score(y_test, preds_xgb))
    print("\nClassification Report of XGBoost:\n", classification_report(y_test, preds_xgb))

    # Compute and plot confusion matrix
    cm_xgb = confusion_matrix(y_test, preds_xgb)
    ConfusionMatrixDisplay(cm_xgb).plot()
    plt.title("Confusion Matrix - XGBoost")
    plt.show()

    # To check the model does not overfit
    # Predict on the training set
    proba_xgb_train = xgb_cl.predict_proba(X_train)
    preds_xgb_train = proba_xgb_train.argmax(axis=1)
    acc_xgb_train = accuracy_score(y_train, preds_xgb_train)
    print("XGBoost Training Accuracy:", acc_xgb_train)
    print("\nXGBoost Training Classification Report:")
    print(classification_report(y_train, preds_xgb_train))

    # Compute learning curve
    train_sizes_xgb, train_scores_xgb, val_scores_xgb = learning_curve(xgb_cl, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

    plt.plot(train_sizes_xgb, train_scores_xgb.mean(axis=1), label="Training")
    plt.plot(train_sizes_xgb, val_scores_xgb.mean(axis=1), label="Validation")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve - XGBoost")
    plt.legend()
    plt.grid()
    plt.show()