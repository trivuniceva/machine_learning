import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
from sklearn.impute import KNNImputer


# Učitavanje podataka
data = pd.read_csv('train.csv')

# Proveravanje nedostajućih vrednosti
print(data.isnull().sum())

# Popunjavanje nedostajućih vrednosti
# imputer = SimpleImputer(strategy='mean')
# data_imputed = imputer.fit_transform(data.drop(columns=['region']))
# data_imputed = pd.DataFrame(data_imputed, columns=data.columns[1:])

# KNN imputacija za popunjavanje nedostajućih vrednosti
knn_imputer = KNNImputer(n_neighbors=5)
data_imputed = knn_imputer.fit_transform(data.drop(columns=['region']))
data_imputed = pd.DataFrame(data_imputed, columns=data.columns[1:])


# Dodavanje kolone 'region' nazad
data_imputed['region'] = data['region']

# Proveravanje rezultata popunjavanja
print(data_imputed.isnull().sum())

# Normalizacija podataka
scaler = StandardScaler()
features = data_imputed.drop(columns=['region'])
features_scaled = scaler.fit_transform(features)

# Mape regiona u brojeve za evaluaciju
regions = data_imputed['region']
region_map = {region: idx for idx, region in enumerate(regions.unique())}
regions_num = regions.map(region_map)


# Treniranje GMM modela
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(features_scaled)

# Predviđanje klastera za trening podatke
predicted_clusters = gmm.predict(features_scaled)

# Evaluacija korišćenjem v measure score
v_measure = v_measure_score(regions_num, predicted_clusters)
print(f'V Measure Score: {v_measure}')

