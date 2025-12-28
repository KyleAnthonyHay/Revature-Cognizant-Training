import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 0: Read data from CSV file
# Make sure customers.csv is in the same folder as this script
data = pd.read_csv('customers.csv')

# Keep only the two columns we care about
# (names must match the CSV header)
customers = data[['annual_income', 'spending_score']].values

# Step 1: Scale features
scaler = StandardScaler()
customers_scaled = scaler.fit_transform(customers)

# Step 2: Determine optimal K using Elbow Method
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(customers_scaled)
    wcss.append(kmeans.inertia_)

# Optional: plot the elbow curve
plt.plot(K_range, wcss, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method for Optimal K')
plt.savefig('elbow_method.png')
plt.show()

# Step 3: Fit K-Means with chosen K (for example, K=4)
K = 6
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
clusters = kmeans.fit_predict(customers_scaled)

# Step 4: Analyze results
print("Cluster Assignments:")
for i in range(K):
    cluster_members = customers[clusters == i]
    print(f"\nCluster {i}:")
    print(f"  Size: {len(cluster_members)} customers")
    print(f"  Avg Income: ${cluster_members[:, 0].mean()*1000:,.0f}")
    print(f"  Avg Spending Score: {cluster_members[:, 1].mean():.1f}")

# Step 5: Get centroids (in original scale)
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
print("\nCluster Centroids (Income, Spending Score):")
for i, centroid in enumerate(centroids_original):
    print(f"  Cluster {i}: Income=${centroid[0]*1000:,.0f}, Score={centroid[1]:.1f}")

# Step 6: Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(customers[:, 0], customers[:, 1], c=clusters, cmap='viridis', s=50, label='Customers')
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.savefig('customer_segments.png')
plt.show()

 