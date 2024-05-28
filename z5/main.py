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

train_path = sys.argv[1]
# test_path = sys.argv[2]

train_data = pd.read_csv(train_path)




# Drop non-numeric columns and handle missing values
# numeric_data = train_data.select_dtypes(include=[np.number])
# imputer = SimpleImputer(strategy='mean')
# numeric_data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)

# Function to detect outliers using IQR method
# def detect_outliers_iqr(data):
#     outliers = {}
#     for column in data.columns:
#         Q1 = data[column].quantile(0.25)
#         Q3 = data[column].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         outliers[column] = data[(data[column] < lower_bound) | (data[column] > upper_bound)][column]
#     return outliers

# # Detect outliers
# outliers = detect_outliers_iqr(numeric_data_imputed)
#
# # Visualize the outliers using box plots
# plt.figure(figsize=(15, 10))
# for i, column in enumerate(numeric_data_imputed.columns, 1):
#     plt.subplot(len(numeric_data_imputed.columns) // 3 + 1, 3, i)
#     sns.boxplot(y=numeric_data_imputed[column])
#     plt.title(f'Box plot of {column}')
#
# plt.tight_layout()
# plt.show()
#
# # Visualize outliers on scatter plots to highlight them
# plt.figure(figsize=(15, 10))
# for i, column in enumerate(numeric_data_imputed.columns, 1):
#     plt.subplot(len(numeric_data_imputed.columns) // 3 + 1, 3, i)
#     sns.scatterplot(data=numeric_data_imputed, x=range(numeric_data_imputed.shape[0]), y=column, label='Data')
#     if not outliers[column].empty:
#         sns.scatterplot(x=outliers[column].index, y=outliers[column], color='red', label='Outliers', marker='o')
#     plt.title(f'Scatter plot of {column} with Outliers')
#     plt.legend()
#
# plt.tight_layout()
# plt.show()
#
#
# numeric_data = train_data.select_dtypes(include=[np.number])
# imputer = SimpleImputer(strategy='mean')
# numeric_data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)

# Compute the correlation matrix
correlation_matrix = numeric_data_imputed.corr()

# Visualize the correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
# plt.title('Correlation Matrix')
# plt.show()

