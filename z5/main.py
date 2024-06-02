import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy import stats
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import DBSCAN


train_path = sys.argv[1]
test_path = sys.argv[2]

data = pd.read_csv(train_path)

knn_imputer = KNNImputer(n_neighbors=5)
data_imputed = knn_imputer.fit_transform(data.drop(columns=['region']))
data_imputed = pd.DataFrame(data_imputed, columns=data.columns[1:])

data_imputed['region'] = data['region']

# Pretvaranje regiona u numeričke vrednosti
region_mapping = {'europe': 0, 'africa': 1, 'asia': 2, 'americas': 3}
data_imputed['region'] = data_imputed['region'].map(region_mapping)

scaler = StandardScaler()
features = data_imputed.drop(columns=['region'])
features_scaled = scaler.fit_transform(features)

scaler = StandardScaler()
features = data_imputed.drop(columns=['region'])
features_scaled = scaler.fit_transform(features)

# z_scores = np.abs(stats.zscore(features_scaled))
# outliers = np.where(z_scores > 3)

dbscan = DBSCAN()
outliers_dbscan = dbscan.fit_predict(features_scaled)
outliers = np.where(outliers_dbscan == -1)


# plt.figure(figsize=(10, 7))
# pca = PCA(n_components=2)
# pca_components = pca.fit_transform(features_scaled)
# plt.scatter(pca_components[:, 0], pca_components[:, 1], c=data_imputed['region'], cmap='viridis', s=50, alpha=0.7)
# plt.xlabel('PCA1')
# plt.ylabel('PCA2')
# plt.title('Originalni podaci')
# plt.colorbar(label='Region')
# plt.show()

# plt.figure(figsize=(10, 7))
# pca_components_outliers = pca.transform(features_scaled[outliers[0]])
# plt.scatter(pca_components[:, 0], pca_components[:, 1], c=data_imputed['region'], cmap='viridis', s=50, alpha=0.7, label='Inliers')
# plt.scatter(pca_components_outliers[:, 0], pca_components_outliers[:, 1], c='red', s=100, alpha=0.7, label='Outliers')
# plt.xlabel('PCA1')
# plt.ylabel('PCA2')
# plt.title('Podaci sa outlier-ima')
# plt.colorbar(label='Region')
# plt.legend()
# plt.show()

features_cleaned = np.delete(features_scaled, outliers[0], axis=0)
data_cleaned = data_imputed.drop(data_imputed.index[outliers[0]])

# plt.figure(figsize=(10, 7))
# pca_components_cleaned = pca.transform(features_cleaned)
# plt.scatter(pca_components_cleaned[:, 0], pca_components_cleaned[:, 1], c=data_cleaned['region'], cmap='viridis', s=50, alpha=0.7)
# plt.xlabel('PCA1')
# plt.ylabel('PCA2')
# plt.title('Podaci nakon uklanjanja outlier-a')
# plt.colorbar(label='Region')
# plt.show()


pca = PCA(n_components=4)
features_pca = pca.fit_transform(features_cleaned)

X_train, X_val, y_train, y_val = train_test_split(features_cleaned, data_cleaned['region'], test_size=0.2, random_state=42)

gmm_params = {
    'n_components': [4, 5, 6, 7],
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
    'random_state': [42]
}

gmm = GaussianMixture()
clf = GridSearchCV(gmm, gmm_params, cv=5, scoring='v_measure_score')
clf.fit(X_train, y_train)

best_gmm = clf.best_estimator_
y_val_pred = best_gmm.predict(X_val)

v_measure = v_measure_score(y_val, y_val_pred)
print(f'V Measure Score on Validation Data: {v_measure}')

test_data = pd.read_csv(test_path)

test_imputed = knn_imputer.transform(test_data.drop(columns=['region']))
test_imputed = pd.DataFrame(test_imputed, columns=test_data.columns[1:])
test_imputed['region'] = test_data['region']

test_features = test_imputed.drop(columns=['region'])
test_features_scaled = scaler.transform(test_features)

z_scores_test = np.abs(stats.zscore(test_features_scaled))
outliers_test = np.where(z_scores_test > 3)
test_features_cleaned = np.delete(test_features_scaled, outliers_test[0], axis=0)
test_cleaned = test_imputed.drop(test_imputed.index[outliers_test[0]])

y_test_pred = best_gmm.predict(test_features_cleaned)

v_measure_test = v_measure_score(test_cleaned['region'], y_test_pred)
print(v_measure_test)