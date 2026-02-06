# Random Forest Model Prediction

import joblib
from datetime import datetime
import pandas as pd


X_master = pd.read_csv("../data/processed/master_features.csv")
tracking = pd.read_csv("../data/processed/tracking_labels.csv")
loyalty_numbers = tracking["Loyalty Number"].copy()
y = tracking["Churn"]


rf_features = [
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

X = X_master[rf_features].copy()

print("=== FEATURES SELECTED ===")
print(f"Total features: {X.columns}")


categorical_cols = [
    "Gender",
    "Education",
    "Marital Status",
    "Loyalty Card",
    "Enrollment Type",
    "Province",
]
X_all_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

rf_model = joblib.load("../models/churn_model.pkl")
predictions = rf_model.predict_proba(X_all_encoded)[:, 1]


# Create risk categories based on churn probability
def categorize_risk(prob):
    if prob >= 0.7:
        return "Critical"
    elif prob >= 0.5:
        return "High"
    elif prob >= 0.35:
        return "Medium"
    else:
        return "Low"


risk_categories_all = [categorize_risk(prob) for prob in predictions]

# Create Dataframe with Predictions and Probabilities

predictions_df = pd.DataFrame(
    {
        "Loyalty Number": loyalty_numbers,
        "Churn Risk Score": predictions.round(4),
        "Churn Risk %": (predictions * 100).round(2),
        "Risk Category": risk_categories_all,
        "Prediction Date": datetime.now().strftime("%Y-%m-%d"),
    }
)

if predictions_df.count().nunique() != 1:
    print("Error: Mismatching Number of Values!")
else:
    print("Perfect File Created!")

print("===== Predictions File =====")
print("Value      |      Count")
print(predictions_df.count())

predictions_df.head(10)


predictions_df.to_csv("../predictions/churn_predictions.csv")
