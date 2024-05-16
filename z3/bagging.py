import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
import sys as sys

train = pd.read_csv(sys.argv[1])
test_preview = pd.read_csv(sys.argv[2])

# nedostajuce vrednosti
# print(train.isnull().sum())
train = train.dropna()

# kodiranje kategorickih atributa
label_encoder = LabelEncoder()

train['bracni_status'] = label_encoder.fit_transform(train['bracni_status'])
train['rasa'] = label_encoder.fit_transform(train['rasa'])
train['tip_posla'] = label_encoder.fit_transform(train['tip_posla'])
train['zdravstveno_osiguranje'] = label_encoder.fit_transform(train['zdravstveno_osiguranje'])

X = train.drop(['obrazovanje'], axis=1)
y = train['obrazovanje']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

micro_f1 = f1_score(y_test, y_pred, average='macro')
print(micro_f1)
