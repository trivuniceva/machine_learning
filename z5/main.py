import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import sys

train_path = sys.argv[1]
# test_path = sys.argv[2]

data = pd.read_csv(train_path)

print(data)



