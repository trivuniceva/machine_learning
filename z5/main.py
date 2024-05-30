import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer


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



