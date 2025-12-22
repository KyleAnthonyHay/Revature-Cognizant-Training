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

print("\n" + "=" * 50)
print("APPROPRIATE K SELECTION WITH JUSTIFICATION:")
print("=" * 50)
print("We chose K=5 because after observing the graph, the number of clusters is not increasing significantly after K=5.")

print("\n" + "=" * 50)
print("BUSINESS-RELEVANT RECOMMENDATIONS:")
print("=" * 50)
for i in range(K):
    cluster_members = customers[clusters == i]
    avg_income = cluster_members[:, 0].mean()
    avg_spending = cluster_members[:, 1].mean()
    segment_name = segment_names[i]
    
    if i == 1:
        print(f"\nCluster {i}: {segment_name} (High income ~${avg_income:,.0f}k, High spending ~{avg_spending:.1f})")
        print("- Channel: VIP email, exclusive events")
        print("- Offer: Premium products, early access, loyalty rewards")
    elif i == 2:
        print(f"\nCluster {i}: {segment_name} (Low income ~${avg_income:,.0f}k, High spending ~{avg_spending:.1f})")
        print("- Channel: Social media (Instagram/TikTok), mobile push")
        print("- Offer: Flash sales, limited-time deals, payment plans")
    elif i == 4:
        print(f"\nCluster {i}: {segment_name} (Low income ~${avg_income:,.0f}k, Low spending ~{avg_spending:.1f})")
        print("- Channel: Email, SMS for sales")
        print("- Offer: Deep discounts, clearance, budget products")

print("\n" + "=" * 50)
print("MEANINGFUL CLUSTER INTERPRETATION:")
print("=" * 50)
for i in range(K):
    cluster_members = customers[clusters == i]
    avg_income = cluster_members[:, 0].mean()
    avg_spending = cluster_members[:, 1].mean()
    segment_name = segment_names[i]
    
    print(f"\nCluster {i}: {segment_name}")
    print(f"- Characteristics: Income ~${avg_income:,.0f}k, Spending ~{avg_spending:.1f}")
    
    if i == 0:
        print("- Real-world meaning: These customers are moderate spenders and are likely to spend a decent amount of money.")
    elif i == 1:
        print("- Real-world meaning: These customers are high spenders and are likely to spend a lot of money.")
    elif i == 2:
        print("- Real-world meaning: These customers are irresponsible spenders and are likely to spend a lot of money.")
    elif i == 3:
        print("- Real-world meaning: These customers are saving-focused customers and are likely to spend a little money.")
    elif i == 4:
        print("- Real-world meaning: These customers are low income customers and are not very likely to spend a lot of money.")