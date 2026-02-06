# Random Forest Prediction Model

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


X_master = pd.read_csv("../data/processed/master_features.csv")
tracking = pd.read_csv("../data/processed/tracking_labels.csv")
y = tracking["Churn"]


# Select Features for Prediction Model


ranfor_features = [
    "Gender",
    "Education",
    "Marital Status",
    "Salary",
    "Loyalty Card",
    "CLV",
    "Enrollment Type",
    "Province",
    "Total Flights",
    "Distance",
    "Points Accumulated",
    "Points Redeemed",
    "Dollar Cost Points Redeemed",
    "Activity 1 Month Before",
    "Activity 2 Months Before",
    "Activity 3 Months Before",
    "Points Most Recent",
    "Overall Trend",
    "Avg Monthly Points",
    "Activity Volatility",
    "Customer Age (Years)",
]

X = X_master[ranfor_features].copy()

print("=== FEATURES SELECTED ===")
print(f"Total features: {X.columns}")


#  We need to encode categorical columns

categorical_cols = [
    "Gender",
    "Education",
    "Marital Status",
    "Loyalty Card",
    "Enrollment Type",
    "Province",
]
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Splitting dataset into a training set and testing set

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print("=== TRAIN/TEST SPLIT ===")
print(f"Training set: {X_train.shape[0]} samples (80%)")
print(f"Test set: {X_test.shape[0]} samples (20%)")
print(f"\nTrain churn rate: {y_train.mean():.2%}")
print(f"Test churn rate: {y_test.mean():.2%}")


# Training Model

rf_model_v2 = RandomForestClassifier(
    n_estimators=150,  # More trees for stability (was 100)
    max_depth=8,  # Shallower trees (was 10)
    min_samples_split=40,  # Require more samples to split (was 20)
    min_samples_leaf=20,  # Larger leaf nodes (was 10)
    max_features="sqrt",  # Limit features per split (NEW)
    class_weight="balanced",  # Keep this
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("Training new model...")
rf_model_v2.fit(X_train, y_train)

# Predict
y_train_pred_v2 = rf_model_v2.predict(X_train)
y_train_proba_v2 = rf_model_v2.predict_proba(X_train)[:, 1]
y_test_pred_v2 = rf_model_v2.predict(X_test)
y_test_proba_v2 = rf_model_v2.predict_proba(X_test)[:, 1]


# New Model (v2)
print("\n--- MODEL V2 (Reduced Overfitting) ---")
print(f"Training Recall:  {recall_score(y_train, y_train_pred_v2):.4f}")
print(f"Test Recall:      {recall_score(y_test, y_test_pred_v2):.4f}")
print(
    f"Gap:              {recall_score(y_train, y_train_pred_v2) - recall_score(y_test, y_test_pred_v2):.4f}"
)
print(f"\nTest Precision:   {precision_score(y_test, y_test_pred_v2):.4f}")
print(f"Test F1:          {f1_score(y_test, y_test_pred_v2):.4f}")
print(f"Test ROC-AUC:     {roc_auc_score(y_test, y_test_proba_v2):.4f}")


# Optimize Threshold Number for Better Predictions

optimal_threshold_v2 = 0.35
y_test_pred_final = (y_test_proba_v2 >= optimal_threshold_v2).astype(int)

# Evaluate
print(f"\nModel V2 + Threshold {optimal_threshold_v2}:")
print(f"Precision: {precision_score(y_test, y_test_pred_final):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred_final):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_test_pred_final):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_test_proba_v2):.4f}")

# Confusion matrix
cm_final = confusion_matrix(y_test, y_test_pred_final)
tn, fp, fn, tp = cm_final.ravel()

print("\n--- FINAL RESULTS ---")
print(f"Churners caught: {tp} out of 366 ({tp / 366 * 100:.1f}%)")
print(f"Churners missed: {fn} ({fn / 366 * 100:.1f}%)")
print(f"Customers flagged: {tp + fp}")
print(f"False alarms: {fp}")

# Export Model for Prediction in Production

joblib.dump(rf_model_v2, "../models/churn_model.pkl")
print("==================================")
print("Model Exporte Succesfully!")
print("==================================")
