
from json import detect_encoding
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import v_measure_score
import numpy as np


def detect_outliers(df, features):
    outlier_indices = []
    
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    
    return list(set(outlier_indices))

def detect_outliers_z_score(df, features, threshold=3):
    outlier_indices = []
    
    for col in features:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        outlier_list_col = df[(z_scores > threshold) | (z_scores < -threshold)].index
        outlier_indices.extend(outlier_list_col)
    
    return list(set(outlier_indices))


data = pd.read_csv('train.csv')

print(data.head())


features = ['Population', 'GDP per Capita', 'Urban Population', 'Life Expectancy', 'Surface Area', 'Literacy Rate']

print(data[features])

print("- - - - ")

print(data[features].isnull().sum())


data[features] = data[features].fillna(data[features].mean())


print(data[features].isnull().sum())

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])
scaled_df = pd.DataFrame(scaled_features, columns=features)
print(scaled_df.head())


# Primena klaster analize
kmeans = KMeans(n_clusters=4, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_features)

# Prikazivanje rezultata klasterovanja
print(data[['region', 'Year', 'Cluster']])

# Primena PCA za smanjenje dimenzionalnosti
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

# Plotovanje rezultata
plt.figure(figsize=(10, 7))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=data['Cluster'], cmap='viridis', s=50, alpha=0.7)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA plot podataka')
plt.colorbar(label='Klaster')
plt.show()


# Izračunavanje matrice korelacije
correlation_matrix = data[features].corr()

# Plotovanje matrice korelacije
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Matrica Korelacije Atributa')
plt.show()

# outliers_indices = detect_outliers(data, features)
outliers_indices = detect_outliers_z_score(data, features)
print("Detektovani outlier-i:", outliers_indices)

# Plotovanje outlier-a na PCA grafiku
plt.figure(figsize=(10, 7))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=data['Cluster'], cmap='viridis', s=50, alpha=0.7)
plt.scatter(pca_components[outliers_indices, 0], pca_components[outliers_indices, 1], color='red', s=100, marker='x', label='Outlier-i')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA plot podataka sa outlier-ima')
plt.colorbar(label='Klaster')
plt.legend()
plt.show()

# Uklanjanje outlier-a
data_without_outliers = data.drop(outliers_indices)

# Prikazivanje rezultata bez outlier-a
print("Podaci bez outlier-a:")
print(data_without_outliers.head())

# Ponovno primena PCA na podacima bez outlier-a
scaled_features_without_outliers = scaler.fit_transform(data_without_outliers[features])
pca_components_without_outliers = pca.fit_transform(scaled_features_without_outliers)

# Plotovanje rezultata bez outlier-a
plt.figure(figsize=(10, 7))
plt.scatter(pca_components_without_outliers[:, 0], pca_components_without_outliers[:, 1], c=data_without_outliers['Cluster'], cmap='viridis', s=50, alpha=0.7)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA plot podataka bez outlier-a')
plt.colorbar(label='Klaster')
plt.show()


# Primena GMM modela
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(scaled_features)

# Predviđanje klastera
clusters = gmm.predict(scaled_features)

# Izračunavanje v mere
true_labels = data['region']  # Pretpostavljamo da region predstavlja stvarne klaster oznake
v_measure = v_measure_score(true_labels, clusters)

print("V Measure Score:", v_measure)