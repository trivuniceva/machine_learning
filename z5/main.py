import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
from sklearn.cluster import KMeans


train_path = sys.argv[1]
# test_path = sys.argv[2]

train_data = pd.read_csv(train_path)


def detect_outliers_zscore(df, features, threshold=3):
    outlier_indices = []

    for col in features:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        outlier_list_col = df[(z_scores > threshold) | (z_scores < -threshold)].index
        outlier_indices.extend(outlier_list_col)

    return list(set(outlier_indices))


data = pd.read_csv(train_path)

print(data.isnull().sum())

knn_imputer = KNNImputer(n_neighbors=5)
data_imputed = knn_imputer.fit_transform(data.drop(columns=['region']))
data_imputed = pd.DataFrame(data_imputed, columns=data.columns[1:])

data_imputed['region'] = data['region']

print(data_imputed.isnull().sum())

scaler = StandardScaler()
features = data_imputed.drop(columns=['region'])
features_scaled = scaler.fit_transform(features)

print(features_scaled)

kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)


pca = PCA(n_components=2)
pca_components = pca.fit_transform(features_scaled)

# Plotovanje rezultata
plt.figure(figsize=(10, 7))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=data['Cluster'], cmap='viridis', s=50, alpha=0.7)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA plot podataka')
plt.colorbar(label='Klaster')
plt.show()


outlier_indices = detect_outliers_zscore(data_imputed, features.columns)
print("Outlier indices:", outlier_indices)

plt.figure(figsize=(10, 7))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=data['Cluster'], cmap='viridis', s=50, alpha=0.7)
plt.scatter(pca_components[outlier_indices, 0], pca_components[outlier_indices, 1], color='red', s=100, marker='x', label='Outlier-i')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA plot podataka sa outlier-ima')
plt.colorbar(label='Klaster')
plt.legend()
plt.show()

data_without_outliers = data.drop(outlier_indices)

print("Podaci bez outlier-a:")
print(data_without_outliers.head())
