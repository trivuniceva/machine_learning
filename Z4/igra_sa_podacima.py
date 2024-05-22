import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, \
    StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import sys


train_path = sys.argv[1]
test_path = sys.argv[2]

data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

label_encoder = LabelEncoder()
label_encoder.fit(data['Genre'])
data['Genre'] = label_encoder.transform(data['Genre'])
test_data['Genre'] = label_encoder.transform(test_data['Genre'])

label_encoder_platform = LabelEncoder()
label_encoder_platform.fit(data['Gaming_Platform'])
data['Gaming_Platform'] = label_encoder_platform.transform(data['Gaming_Platform'])
test_data['Gaming_Platform'] = label_encoder_platform.transform(test_data['Gaming_Platform'])


# imputer = SimpleImputer(strategy='mean')
# data_filled = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
# test_filled = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)

# Kreiranje KNNImputer objekta
knn_imputer = KNNImputer(n_neighbors=5)

# Popunjavanje nedostajućih vrednosti u trening podacima
data_filled = pd.DataFrame(knn_imputer.fit_transform(data), columns=data.columns)
test_filled = pd.DataFrame(knn_imputer.transform(test_data), columns=test_data.columns)

# n_features = data_filled.shape[1] - 1
# n_components = min(7, n_features)
#
# pca = PCA(n_components=n_components)
# data_reduced = pca.fit_transform(data_filled.drop(columns='Genre'))
# test_reduced = pca.transform(test_filled.drop(columns='Genre'))

pca = PCA()

data_reduced = pca.fit_transform(data_filled.drop(columns='Genre'))
test_reduced = pca.transform(test_filled.drop(columns='Genre'))

explained_variance = pca.explained_variance_ratio_.cumsum()
n_components = np.argmax(explained_variance >= 0.95) + 1

pca = PCA(n_components=n_components)
data_reduced = pca.fit_transform(data_filled.drop(columns='Genre'))
test_reduced = pca.transform(test_filled.drop(columns='Genre'))

# - - - - - - - - - - - - - - - - - - - - - - - - -
X_train = data_reduced
y_train = data_filled['Genre']

X_test = test_reduced
y_test = test_filled['Genre']

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred_test = rf_classifier.predict(X_test)
f1_macro_test = f1_score(y_test, y_pred_test, average='macro')

print(f1_macro_test)


# def evaluate_model(model, X_train, y_train, X_test, y_test):
#     model.fit(X_train, y_train)
#     y_pred_test = model.predict(X_test)
#     f1_macro_test = f1_score(y_test, y_pred_test, average='macro')
#     return f1_macro_test
#
# # Voting
# voting_model = VotingClassifier(estimators=[
#     ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
#     ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
#     ('svc', SVC(kernel='linear', probability=True))
# ], voting='soft')
# f1_voting = evaluate_model(voting_model, X_train, y_train, X_test, y_test)
# print(f'Voting F1 Score: {f1_voting}')
