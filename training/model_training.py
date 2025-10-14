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
# df_v1 = pd.read_csv("./synthetic_patient_basic_setting_v1.csv")
# df_v2 = pd.read_csv("./synthetic_patient_basic_setting_v2.csv")

# Upload the binned version of the dataset
df_v2 = pd.read_csv("./synthetic_patient_basic_setting_v2_binned.csv")

# Define X and y
y = df_v2['disease']
X = df_v2.drop(columns=['disease', 'patient_id', 'disease_group'])

# To display all variables
print(f"Variables: {list(X.columns)}")
# To display the values the target variable can take
print("\nDifferent classes for target y:", y.unique(), "\n")

# Convert categorical target variable to numerical: it is needed for XGBoost
y_cat = y.astype('category')       # Data type of elements of y becomes 'category' (order is defined by order of categories)
codes = y_cat.cat.codes            # Each integer code represents the category of the corresponding element observation
categories = y_cat.cat.categories  # Index of category labels

# To check which category is mapped to which code
for code, category in enumerate(categories):
    print("Category:", category, "-> Code:", code)

if y.dtype == 'O':
    y = y.astype('category').cat.codes  # maps categories to 0,...,n-1

# Split the data: we want training 65%, testing 15%, pool for anchors 10%, set for conformal predictions 10%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.65, random_state=42, stratify=y)
X_test, X_temp2, y_test, y_temp2 = train_test_split(X_temp, y_temp, train_size=0.4286, random_state=42, stratify=y_temp)
X_anchors, X_conf_pred, y_anchors, y_conf_pred = train_test_split(X_temp2, y_temp2, train_size=0.5, random_state=42, stratify=y_temp2)

print(f"\nTraining set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Testing set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"Anchors set: {len(X_anchors)} samples ({len(X_anchors)/len(X)*100:.1f}%)")
print(f"Conformal predictions set: {len(X_conf_pred)} samples ({len(X_conf_pred)/len(X)*100:.1f}%)")
print(f"\nTotal: {len(X_train) + len(X_test) + len(X_anchors) + len(X_conf_pred)} samples")


"""Random Forest Classifier"""
# Initialize the model
rf = RandomForestClassifier(
    n_estimators=100,   # number of trees
    random_state=42,
    min_samples_leaf=5,
    max_depth=10,
    n_jobs=-1           # use all CPU cores
)

# Fit the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf.predict(X_test)

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


"""XGBoost classifier"""
classes = np.unique(y)
num_classes = len(classes)

# Initialize the model
xgb_cl = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=num_classes,
    eval_metric="mlogloss",          # primary optimization metric
    n_estimators=200,
    min_child_weight=5,
    reg_alpha=0.1,                   # L1 regularization
    reg_lambda=0.1,                  # L2 regularization
    learning_rate=0.05,
    max_depth=5,
    tree_method="hist",
    random_state=42,
)

# Fit the model
xgb_cl.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Predict on the test set
proba_xgb = xgb_cl.predict_proba(X_test)  
preds_xgb = proba_xgb.argmax(axis=1)

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