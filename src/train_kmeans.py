# KMeans Clutering Model

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

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

# Scale features
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Convert back to DataFrame for easier handling
X_cluster_scaled_df = pd.DataFrame(
    X_cluster_scaled, columns=kmeans_features, index=X_cluster.index
)

print("\nFeatures scaled...")


# Testing different number of clusters

k_range = range(2, 11)  # Test 2 to 10 clusters
inertias = []
silhouette_scores = []
davies_bouldin_scores = []

print("\nTesting different cluster counts...")

for k in k_range:
    # Fit K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)

    # Calculate metrics
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_cluster_scaled, kmeans.labels_)
    db_score = davies_bouldin_score(X_cluster_scaled, kmeans.labels_)

    inertias.append(inertia)
    silhouette_scores.append(silhouette)
    davies_bouldin_scores.append(db_score)


optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]

print(f"\nOptimal Number of Clusters: {optimal_k}")


# Training Clustering Model

optimal_k = 3

# Train final model
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10, max_iter=300)

print("\nTraining K-Means model...")
cluster_labels = kmeans_final.fit_predict(X_cluster_scaled)

print("Model trained successfully")
print(f"   Clusters: {optimal_k}")
print(f"   Iterations: {kmeans_final.n_iter_}")
print(f"   Inertia: {kmeans_final.inertia_:.0f}")

# Add cluster labels to original data
X_cluster["Cluster"] = cluster_labels
X_master["Cluster"] = cluster_labels

# Cluster distribution
print("\n=== CLUSTER DISTRIBUTION ===")
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
print(cluster_counts)
print("\nCluster sizes:")
for cluster_id, count in cluster_counts.items():
    print(
        f"  Cluster {cluster_id}: {count:,} customers ({count / len(cluster_labels) * 100:.1f}%)"
    )


# Calculate cluster centers (in original scale)
cluster_centers_scaled = kmeans_final.cluster_centers_
cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)

cluster_profiles = pd.DataFrame(cluster_centers_original, columns=kmeans_features)

# Add cluster size
cluster_profiles["Cluster Size"] = cluster_counts.values
cluster_profiles["Cluster"] = range(optimal_k)

# Add churn rate per cluster
cluster_churn_rate = []
for i in range(optimal_k):
    cluster_mask = cluster_labels == i
    churn_rate = churn_labels[cluster_mask].mean()
    cluster_churn_rate.append(churn_rate)

cluster_profiles["Churn Rate"] = cluster_churn_rate

# Detailed view for each cluster

for cluster_id in range(optimal_k):
    print(f"\n--- CLUSTER {cluster_id} ---")
    cluster_mask = cluster_labels == cluster_id
    cluster_data = X_cluster[cluster_mask]

    print(
        f"Size: {cluster_mask.sum():,} customers ({cluster_mask.sum() / len(cluster_labels) * 100:.1f}%)"
    )
    print(f"Churn Rate: {cluster_churn_rate[cluster_id]:.2%}")

    print("\nKey Characteristics:")
    print(f"  Avg Salary: ${cluster_data['Salary'].mean():,.0f}")
    print(f"  Avg CLV: ${cluster_data['CLV'].mean():,.0f}")
    print(f"  Avg Total Flights: {cluster_data['Total Flights'].mean():.1f}")
    print(f"  Avg Distance: {cluster_data['Distance'].mean():,.0f} km")
    print(f"  Avg Points Accumulated: {cluster_data['Points Accumulated'].mean():,.0f}")
    print(
        f"  Avg Customer Age: {cluster_data['Customer Age (Years)'].mean():.1f} years"
    )


joblib.dump(kmeans_final, "../models/clustering_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
