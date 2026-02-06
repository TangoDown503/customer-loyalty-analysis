# Cluster Creation

import joblib
from datetime import datetime
import pandas as pd


# Load the master data
X_master = pd.read_csv("../data/processed/master_features.csv")
tracking = pd.read_csv("../data/processed/tracking_labels.csv")

# Keep loyalty numbers and churn for analysis later
loyalty_numbers = tracking["Loyalty Number"].copy()
churn_labels = tracking["Churn"].copy()

# Focus on behavioral and value-based features
kmeans_features = [
    "Salary",
    "CLV",
    "Total Flights",
    "Distance",
    "Points Accumulated",
    "Points Redeemed",
    "Dollar Cost Points Redeemed",
    "Points Most Recent",
    "Avg Monthly Points",
    "Activity Volatility",
    "Overall Trend",
    "Customer Age (Years)",
]

# Select features
X_cluster = X_master[kmeans_features].copy()

print(f"\nClustering data shape: {X_cluster.shape}")

# Import model and scaler

kmeans_model = joblib.load("../models/clustering_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
print("\nModels Imported Succesfully...")

X_cluster_scaled = scaler.transform(X_cluster)
clusters = kmeans_model.predict(X_cluster_scaled)
print("\nPredictions Created!")

# Create cluster names
cluster_names = {0: "Super User", 1: "Loyal Regular", 2: "Flight Risk"}


cluster_names_list = [cluster_names[c] for c in clusters]


kmeans_predictions_df = pd.DataFrame(
    {
        "Loyalty Number": loyalty_numbers,
        "Cluster ID": clusters,
        "Cluster Name": cluster_names_list,
        "Prediction Date": datetime.now().strftime("%Y-%m-%d"),
    }
)

if kmeans_predictions_df.count().nunique() != 1:
    print("Error: Mismatching Number of Values!")
else:
    print("Perfect File Created!")

print("===== Predictions File =====")
print("Value      |      Count")
print(kmeans_predictions_df.count())

kmeans_predictions_df.head(10)


kmeans_predictions_df.to_csv("../predictions/cluster_predictions.csv")
