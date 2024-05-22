import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import sys

# Učitavanje trening i test podataka iz argumenata komandne linije
train_path = sys.argv[1]
test_path = sys.argv[2]

data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Kodiranje tekstualnih atributa za trening i test podatke
label_encoder = LabelEncoder()
all_genres = pd.concat([data['Genre'], test_data['Genre']], axis=0)
label_encoder.fit(all_genres)

# Provera da li postoje neviđene oznake u test podacima
# unseen_labels = set(test_data['Genre']) - set(data['Genre'])
# if unseen_labels:
#     raise ValueError("Test podaci sadrže neviđene oznake za žanrove: {}".format(unseen_labels))

data['Genre'] = label_encoder.transform(data['Genre'])
test_data['Genre'] = label_encoder.transform(test_data['Genre'])

# Kodiranje tekstualnih atributa za trening i test podatke
label_encoder_platform = LabelEncoder()
all_platforms = pd.concat([data['Gaming_Platform'], test_data['Gaming_Platform']], axis=0)
label_encoder_platform.fit(all_platforms)

# Transformacija platformi
data['Gaming_Platform'] = label_encoder_platform.transform(data['Gaming_Platform'])
test_data['Gaming_Platform'] = label_encoder_platform.transform(test_data['Gaming_Platform'])

# Priprema trening podataka
imputer = SimpleImputer(strategy='mean')
data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Redukcija dimenzionalnosti na trening podacima
pca = PCA(n_components=7)
data_reduced = pca.fit_transform(data_filled)

# Podela trening podataka na X_train i y_train
X_train = data_reduced
y_train = data['Genre']

# Izgradnja ansambla klasifikatora (ovde koristimo Random Forest)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Priprema test podataka
test_filled = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

# Redukcija dimenzionalnosti na test podacima
test_reduced = pca.transform(test_filled)

# Podela test podataka na X_test i y_test
X_test = test_reduced
y_test = test_data['Genre']

# Predikcija na test podacima
y_pred_test = rf_classifier.predict(X_test)

f1_macro_test = f1_score(y_test, y_pred_test, average='macro')

print(f1_macro_test)
