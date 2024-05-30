import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Učitavanje podataka
data = pd.read_csv('train.csv')

# Proveravanje nedostajućih vrednosti
print(data.isnull().sum())

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

# Podela na trening i validacione podatke
X_train, X_val, y_train, y_val = train_test_split(features_scaled, data_imputed['region'], test_size=0.2, random_state=42)

# Treniranje GMM modela na trening podacima
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X_train)

# Predikcija klastera na validacionom skupu
y_val_pred = gmm.predict(X_val)

# Evaluacija modela na validacionom skupu
v_measure = v_measure_score(y_val, y_val_pred)
print(f'V Measure Score on Validation Data: {v_measure}')


# Učitavanje test podataka
test_data = pd.read_csv('test.csv')

# Provera nedostajućih vrednosti u test podacima
print(test_data.isnull().sum())

# KNN imputacija za popunjavanje nedostajućih vrednosti u test podacima
test_imputed = knn_imputer.transform(test_data.drop(columns=['region']))
test_imputed = pd.DataFrame(test_imputed, columns=test_data.columns[1:])

# Dodavanje kolone 'region' nazad
test_imputed['region'] = test_data['region']

# Normalizacija test podataka
test_features = test_imputed.drop(columns=['region'])
test_features_scaled = scaler.transform(test_features)

# Predikcija klastera na test skupu
y_test_pred = gmm.predict(test_features_scaled)

# Evaluacija modela na test skupu
v_measure_test = v_measure_score(test_imputed['region'], y_test_pred)
print(f'V Measure Score on Test Data: {v_measure_test}')

