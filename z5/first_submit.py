import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
from sklearn.model_selection import train_test_split
from scipy import stats
# import matplotlib.pyplot as plt
import sys

train_path = sys.argv[1]
test_path = sys.argv[2]

data = pd.read_csv(train_path)
# print(data.isnull().sum())

knn_imputer = KNNImputer(n_neighbors=5)
data_imputed = knn_imputer.fit_transform(data.drop(columns=['region']))
data_imputed = pd.DataFrame(data_imputed, columns=data.columns[1:])

data_imputed['region'] = data['region']

# print(data_imputed.isnull().sum())

scaler = StandardScaler()
features = data_imputed.drop(columns=['region'])
features_scaled = scaler.fit_transform(features)

z_scores = np.abs(stats.zscore(features_scaled))
outliers = np.where(z_scores > 3)

pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

# plt.figure(figsize=(14, 6))

# plt.subplot(1, 2, 1)
# plt.scatter(features_pca[:, 0], features_pca[:, 1], c='blue', label='Data Points')
# plt.scatter(features_pca[outliers[0], 0], features_pca[outliers[0], 1], c='red', label='Outliers')
# plt.title('Data with Outliers')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend()

features_cleaned = np.delete(features_scaled, outliers[0], axis=0)
data_cleaned = data_imputed.drop(data_imputed.index[outliers[0]])

features_cleaned_pca = pca.transform(features_cleaned)

# plt.subplot(1, 2, 2)
# plt.scatter(features_cleaned_pca[:, 0], features_cleaned_pca[:, 1], c='blue', label='Data Points')
# plt.title('Data without Outliers')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend()

# plt.tight_layout()
# plt.show()

X_train, X_val, y_train, y_val = train_test_split(features_cleaned, data_cleaned['region'], test_size=0.2, random_state=42)

gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X_train)

y_val_pred = gmm.predict(X_val)

v_measure = v_measure_score(y_val, y_val_pred)
# print(f'V Measure Score on Validation Data: {v_measure}')

test_data = pd.read_csv(test_path)

# print(test_data.isnull().sum())

test_imputed = knn_imputer.transform(test_data.drop(columns=['region']))
test_imputed = pd.DataFrame(test_imputed, columns=test_data.columns[1:])

test_imputed['region'] = test_data['region']

test_features = test_imputed.drop(columns=['region'])
test_features_scaled = scaler.transform(test_features)

z_scores_test = np.abs(stats.zscore(test_features_scaled))
outliers_test = np.where(z_scores_test > 3)
test_features_cleaned = np.delete(test_features_scaled, outliers_test[0], axis=0)
test_cleaned = test_imputed.drop(test_imputed.index[outliers_test[0]])

y_test_pred = gmm.predict(test_features_cleaned)

v_measure_test = v_measure_score(test_cleaned['region'], y_test_pred)
print(v_measure_test)
