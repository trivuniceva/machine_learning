import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import sys


train_path = sys.argv[1]
test_path = sys.argv[2]

train_data = pd.read_csv(train_path)

label_encoder = LabelEncoder()
train_data['Gaming_Platform'] = label_encoder.fit_transform(train_data['Gaming_Platform'])
train_data['Genre'] = label_encoder.fit_transform(train_data['Genre'])

imputer = SimpleImputer(strategy='mean')
data_filled = pd.DataFrame(imputer.fit_transform(train_data), columns=train_data.columns)

pca = PCA(n_components=7)
data_reduced = pca.fit_transform(data_filled)

X = data_reduced
y = train_data['Genre']

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y)

y_pred = rf_classifier.predict(X)
f1_macro = f1_score(y, y_pred, average='macro')

print(f1_macro)
