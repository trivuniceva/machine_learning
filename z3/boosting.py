import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
import sys as sys

train = pd.read_csv('train.csv')
test_preview = pd.read_csv('test_preview.csv')

# nedostajuce vrednosti
# print(train.isnull().sum())
train = train.dropna()
train.reset_index(drop=True, inplace=True)

print(train)

label_encoder = LabelEncoder()
train['bracni_status'] = label_encoder.fit_transform(train['bracni_status'])
train['rasa'] = label_encoder.fit_transform(train['rasa'])
train['tip_posla'] = label_encoder.fit_transform(train['tip_posla'])
train['zdravstveno_osiguranje'] = label_encoder.fit_transform(train['zdravstveno_osiguranje'])


X_train = train.drop(['obrazovanje'], axis=1)
y_train = train['obrazovanje']

y_encoder = LabelEncoder()
y_train = y_encoder.fit_transform(y_train)

clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
clf.fit(X_train, y_train)


test_preview = test_preview.dropna()
test_preview.reset_index(drop=True, inplace=True)

test_preview['bracni_status'] = label_encoder.transform(test_preview['bracni_status'])
test_preview['rasa'] = label_encoder.transform(test_preview['rasa'])
test_preview['tip_posla'] = label_encoder.transform(test_preview['tip_posla'])
test_preview['zdravstveno_osiguranje'] = label_encoder.transform(test_preview['zdravstveno_osiguranje'])

X_test = test_preview.drop(['obrazovanje'], axis=1)

y_test = test_preview['obrazovanje']
y_test = y_encoder.transform(y_test)

X_test = X_test[mask]
y_test = y_test[mask]

y_pred = clf.predict(X_test)

micro_f1 = f1_score(y_test, y_pred, average='macro')
print(micro_f1)
