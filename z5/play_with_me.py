import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import v_measure_score
from sklearn.decomposition import PCA

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test2.csv')


def remove_outliers_zscore(df, columns):
    z_scores = np.abs((df[columns] - df[columns].mean()) / df[columns].std())
    outliers = (z_scores > 3).any(axis=1)
    return outliers


# Detekcija i uklanjanje outliera iz trening skupa
columns_to_check = ['Year', 'Population', 'GDP per Capita', 'Urban Population', 'Life Expectancy', 'Surface Area',
                    'Literacy Rate']
train_data_clean, train_data_outliers = remove_outliers(train_data, columns_to_check)

# Plotovanje podataka pre uklanjanja outliera
plot_data(train_data, 'Training Data with Outliers')

# Plotovanje podataka nakon uklanjanja outliera
plot_data(train_data_clean, 'Training Data without Outliers')

# Izdvajanje ciljne promenljive (region) iz trening i testnog skupa
train_labels = train_data_clean['region']
test_labels = test_data['region']
train_data_clean = train_data_clean.drop(columns=['region'])
test_data = test_data.drop(columns=['region'])

# Obrada nedostajućih vrednosti
imputer = SimpleImputer(strategy='mean')
train_data_imputed = imputer.fit_transform(train_data_clean)
test_data_imputed = imputer.transform(test_data)

# Skaliranje podataka
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data_imputed)
test_data_scaled = scaler.transform(test_data_imputed)

# Treniranje GMM modela
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(train_data_scaled)

# Predikcija klastera za test skup
predicted_clusters = gmm.predict(test_data_scaled)


# Funkcija za mapiranje klastera na regione
def map_clusters_to_regions(predicted_clusters, true_labels):
    mapping = {}
    unique_clusters = np.unique(predicted_clusters)
    for cluster in unique_clusters:
        mask = (predicted_clusters == cluster)
        most_common_region = np.unique(true_labels[mask], return_counts=True)[0][
            np.argmax(np.unique(true_labels[mask], return_counts=True)[1])]
        mapping[cluster] = most_common_region
    return [mapping[cluster] for cluster in predicted_clusters]


# Mapiranje klastera na regione
mapped_clusters = map_clusters_to_regions(predicted_clusters, test_labels)

# Evaluacija modela
v_measure = v_measure_score(test_labels, mapped_clusters)
print(f'V-measure score: {v_measure}')
