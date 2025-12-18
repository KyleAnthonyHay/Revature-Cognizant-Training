"""
Exercise 03: K-Means Customer Segmentation
===========================================

SETUP INSTRUCTIONS:
-------------------
1. Create a virtual environment:
   python3 -m venv venv

2. Activate the virtual environment:
   - On macOS/Linux: source venv/bin/activate
   - On Windows: venv\Scripts\activate

3. Install required packages:
   pip install -r requirements.txt

4. Run the program:
   python3 main.py

5. Graphs will open on a new window and saved as customer_segments.png and elbow_method.png

Complete implementation of K-Means clustering for customer segmentation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 0: Read data from CSV
data = pd.read_csv('Mall_Customers.csv')

# Keep only the two columns we care about processing
customers = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

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

plt.plot(K_range, wcss, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method for Optimal K')
plt.savefig('elbow_method.png')
plt.show()

# Step 3: Fit K-Means with chosen K (elbow suggests K=5)
K = 5
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
clusters = kmeans.fit_predict(customers_scaled)

# Segment names dictionary
segment_names = {
    0: "Moderate Spenders",
    1: "High Spenders",
    2: "Irresponsible Spenders",
    3: "Saving-Focused Customers",
    4: "Low Income Customers"
}

# Step 4: Analyze results
print("Cluster Assignments:")
for i in range(K):
    cluster_members = customers[clusters == i]
    segment_name = segment_names[i]
    print(f"\nCluster {i}: {segment_name}")
    print(f"  Size: {len(cluster_members)} customers")
    print(f"  Avg Income: ${cluster_members[:, 0].mean():,.0f}k")
    print(f"  Avg Spending Score: {cluster_members[:, 1].mean():.1f}")

# Step 5: Get centroids (in original scale)
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
print("\nCluster Centroids (Income, Spending Score):")
for i, centroid in enumerate(centroids_original):
    segment_name = segment_names[i]
    print(f"  Cluster {i} ({segment_name}): Income=${centroid[0]:,.0f}k, Score={centroid[1]:.1f}")

# Step 6: Visualize the clusters
plt.figure(figsize=(10, 6))
for i in range(K):
    mask = clusters == i
    plt.scatter(
        customers[mask, 0], 
        customers[mask, 1], 
        label=f'Cluster {i}: {segment_names[i]}',
        alpha=0.6,
        s=50
    )
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.savefig('customer_segments.png')
plt.show()

"""
=============================================
APPROPRIATE K SELECTION WITH JUSTIFICATION:
=============================================
We chose K=5 because after observing the graph, the number of clusters is not increasing significantly after K=5.

==================================
BUSINESS-RELEVANT RECOMMENDATIONS:
==================================
Cluster 1: High Spenders (High income ~$87k, High spending ~82.1) 
- Channel: VIP email, exclusive events
- Offer: Premium products, early access, loyalty rewards

Cluster 2: Irresponsible Spenders (Low income ~$26k, High spending ~79.4)
- Channel: Social media (Instagram/TikTok), mobile push
- Offer: Flash sales, limited-time deals, payment plans

Cluster 4: Low Income Customers (Low income ~$26k, Low spending ~20.9)
- Channel: Email, SMS for sales
- Offer: Deep discounts, clearance, budget products

==================================
MEANINGFUL CLUSTER INTERPRETATION:
==================================
Cluster 0: Moderate Spenders
- Characteristics: Moderate income (~$55k), moderate spending (~49.5)
- Real-world meaning: These customers are moderate spenders and are likely to spend a decent amount of money.

Cluster 1: High Spenders
- Characteristics: High income (~$87K), high spending (~82.1)
- Real-world meaning: These customers are high spenders and are likely to spend a lot of money.

Cluster 2: Irresponsible Spenders
- Characteristics: Low income (~$26k), high spending (~79.4)
- Real-world meaning: These customers are irresponsible spenders and are likely to spend a lot of money.

Cluster 3: Saving-Focused Customers
- Characteristics: High income (~$88k), low spending (~17.1)
- Real-world meaning: These customers are saving-focused customers and are likely to spend a little money.

Cluster 4: Low Income Customers
- Characteristics: Low income (~$26k), low spending (~20.9)
- Real-world meaning: These customers are low income customers and are not very likely to spend a lot of money.
"""