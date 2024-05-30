import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
from sklearn.model_selection import train_test_split
from scipy import stats

data = pd.read_csv('train.csv')

print(data.isnull().sum())

knn_imputer = KNNImputer(n_neighbors=5)
data_imputed = knn_imputer.fit_transform(data.drop(columns=['region']))
data_imputed = pd.DataFrame(data_imputed, columns=data.columns[1:])

data_imputed['region'] = data['region']

print(data_imputed.isnull().sum())

scaler = StandardScaler()
features = data_imputed.drop(columns=['region'])
features_scaled = scaler.fit_transform(features)

z_scores = np.abs(stats.zscore(features_scaled))
outliers = np.where(z_scores > 3)
features_cleaned = np.delete(features_scaled, outliers[0], axis=0)
data_cleaned = data_imputed.drop(data_imputed.index[outliers[0]])








X_train, X_val, y_train, y_val = train_test_split(features_cleaned, data_cleaned['region'], test_size=0.2, random_state=42)

gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X_train)

y_val_pred = gmm.predict(X_val)

v_measure = v_measure_score(y_val, y_val_pred)
print(f'V Measure Score on Validation Data: {v_measure}')
