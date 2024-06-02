import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy import stats
import sys

train_path = sys.argv[1]
test_path = sys.argv[2]

data = pd.read_csv(train_path)

knn_imputer = KNNImputer(n_neighbors=5)
data_imputed = knn_imputer.fit_transform(data.drop(columns=['region']))
data_imputed = pd.DataFrame(data_imputed, columns=data.columns[1:])

data_imputed['region'] = data['region']

scaler = StandardScaler()
features = data_imputed.drop(columns=['region'])
features_scaled = scaler.fit_transform(features)

z_scores = np.abs(stats.zscore(features_scaled))
outliers = np.where(z_scores > 3)

features_cleaned = np.delete(features_scaled, outliers[0], axis=0)
data_cleaned = data_imputed.drop(data_imputed.index[outliers[0]])

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

# Mapiranje klastere na regione
cluster_mapping = {0: 'europe', 1: 'africa', 2: 'asia', 3: 'americas'}
y_test_pred_mapped = [cluster_mapping.get(cluster, 'unknown') for cluster in y_test_pred]

v_measure_test = v_measure_score(test_cleaned['region'], y_test_pred_mapped)
print(v_measure_test)
