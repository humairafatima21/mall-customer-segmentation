# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

plt.rcParams.update({'figure.max_open_warning': 0})

# ── Output folder ─────────────────────────────────────────────
OUTPUT_DIR = "/content/output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(filename):
    """Simpan gambar ke OUTPUT_DIR lalu tampilkan."""
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

# ── Load Dataset ──────────────────────────────────────────────
file_id = "185dkvqgW-v7jPnWalKF1HNnKodUYYJGD"
url = f"https://drive.google.com/uc?id={file_id}"
Mall_Customers = pd.read_csv(url)
print("Loaded dataset shape:", Mall_Customers.shape)
print(Mall_Customers.head())

# ── Pemeriksaan Data ──────────────────────────────────────────
print("\n--- Data checking ---")
print("Missing values per column:\n", Mall_Customers.isnull().sum())
print("Duplicate rows:", Mall_Customers.duplicated().sum())
print("\nDescriptive statistics:")
print(Mall_Customers.describe(include='all').T)

# ── Persiapan Data ────────────────────────────────────────────
Mall_Customers_prep = Mall_Customers.copy()
if 'Gender' in Mall_Customers_prep.columns:
    le = LabelEncoder()
    Mall_Customers_prep['Gender_enc'] = le.fit_transform(Mall_Customers_prep['Gender'])

features = [c for c in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
            if c in Mall_Customers_prep.columns]
print("\nFeatures used for clustering:", features)

X = Mall_Customers_prep[features].dropna().copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Elbow & Silhouette ────────────────────────────────────────
K_range = list(range(2, 11))
inertias, sil_scores = [], []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

best_k_by_sil = K_range[int(np.argmax(sil_scores))]
best_sil = max(sil_scores)
print(f"\nBest k by silhouette: {best_k_by_sil} (silhouette = {best_sil:.3f})")

# ── KMeans Final ──────────────────────────────────────────────
kmeans = KMeans(n_clusters=best_k_by_sil, random_state=42, n_init=50)
labels_final = kmeans.fit_predict(X_scaled)
X['Cluster'] = labels_final

cluster_counts = X['Cluster'].value_counts().sort_index()
cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
centers_df = pd.DataFrame(cluster_centers_original, columns=X.columns[:-1])
centers_df.index.name = 'Cluster'
print("\nCluster counts:\n", cluster_counts.sort_index())
print("\nCluster centers (original scale):")
print(centers_df.round(2))

# ── PCA ───────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
pca2 = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(pca2, columns=['PC1', 'PC2'])
pca_df['Cluster'] = labels_final


#  PLOTS

# 1. age_distribution.png
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(X['Age'], bins=15, color='skyblue', edgecolor='black')
ax.set_title("Age Distribution")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")
plt.tight_layout()
save("age_distribution.png")

# 2. annual_income_distribution.png
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(X['Annual Income (k$)'], bins=15, color='salmon', edgecolor='black')
ax.set_title("Annual Income Distribution")
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Frequency")
plt.tight_layout()
save("annual_income_distribution.png")

# 3. spending_score_distribution.png
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(X['Spending Score (1-100)'], bins=15, color='mediumseagreen', edgecolor='black')
ax.set_title("Spending Score Distribution")
ax.set_xlabel("Spending Score (1-100)")
ax.set_ylabel("Frequency")
plt.tight_layout()
save("spending_score_distribution.png")

# 4. income_vs_spending_clusters.png
fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.tab10.colors
for c in sorted(cluster_counts.index):
    sub = X[X['Cluster'] == c]
    ax.scatter(sub['Annual Income (k$)'], sub['Spending Score (1-100)'],
               label=f'Cluster {c}', alpha=0.7, color=colors[c])
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_title("Customer Segments — Income vs Spending")
ax.legend()
ax.grid(True)
plt.tight_layout()
save("income_vs_spending_clusters.png")

# 5. pca_clusters.png
fig, ax = plt.subplots(figsize=(8, 6))
for c in sorted(cluster_counts.index):
    s = pca_df[pca_df['Cluster'] == c]
    ax.scatter(s['PC1'], s['PC2'], label=f'Cluster {c}', alpha=0.7, color=colors[c])
ax.set_title("PCA Visualization of Clusters")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()
ax.grid(True)
plt.tight_layout()
save("pca_clusters.png")

# 6. elbow_method.png
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(K_range, inertias, marker='o', markersize=8, linewidth=2, color='steelblue')
ax.set_title("Elbow Method")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia")
ax.grid(True)
plt.tight_layout()
save("elbow_method.png")

# 7. silhouette_score.png
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(K_range, sil_scores, marker='o', markersize=8, linewidth=2, color='green')
ax.axvline(best_k_by_sil, linestyle='--', color='red',
           label=f'Best k = {best_k_by_sil} ({best_sil:.3f})')
ax.set_title("Silhouette Score per k")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Silhouette Score")
ax.legend()
ax.grid(True)
plt.tight_layout()
save("silhouette_score.png")

# ── Konfirmasi file tersimpan ─────────────────────────────────
print("\nFile yang tersimpan di", OUTPUT_DIR + ":")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(" ", f)
