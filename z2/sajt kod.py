import sys
import pandas as pd
import numpy as np


class CustomKNN:
    def __init__(self, n_neighbors=5, alpha=0):
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            prediction = np.mean(self.y_train[nearest_indices])

            if self.alpha != 0:
                prediction -= self.alpha * np.linalg.norm(self.y_train[nearest_indices], ord=1)
            predictions.append(prediction)

        return predictions


def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file, delimiter="\t")
    test_data = pd.read_csv(test_file, delimiter="\t")
    return train_data, test_data


def detect_outliers_zscore(data):
    threshold = 3
    outliers = []
    for col in data.columns:
        z_scores = (data[col] - data[col].mean()) / data[col].std()
        outliers.extend(data.index[z_scores.abs() > threshold])
    return list(set(outliers))

def standardize_data(train_data, test_data):
    train_mean = train_data.mean()
    train_std = train_data.std()
    X_train_standardized = (train_data - train_mean) / train_std
    X_test_standardized = (test_data - train_mean) / train_std
    return X_train_standardized, X_test_standardized


def preprocess_data(train_data, test_data):
    # Mapiranje tipa menjača na numeričke vrednosti
    train_data['Menjac'] = train_data['Menjac'].map({'Manuelni': 0, 'Automatski': 1})
    test_data['Menjac'] = test_data['Menjac'].map({'Manuelni': 0, 'Automatski': 1})

    # Kopiranje podataka da biste izbegli SettingWithCopyWarning
    train_data = train_data.copy()
    test_data = test_data.copy()

    # Izdvajanje atributa za treniranje i ciljne promenljive
    X_train = train_data[['Kilometraza', 'Konjske snage', 'Menjac']].copy()
    y_train = train_data['Cena'].copy()
    X_test = test_data[['Kilometraza', 'Konjske snage', 'Menjac']].copy()
    y_test = test_data['Cena'].copy()

    # Uklanjanje redova sa nedostajućim podacima
    X_train.dropna(inplace=True)
    y_train = y_train.loc[X_train.index]
    X_test.dropna(inplace=True)
    y_test = y_test.loc[X_test.index]

    # Detekcija outlier-a
    outliers_train = detect_outliers_zscore(X_train)
    outliers_test = detect_outliers_zscore(X_test)

    # Uklanjanje outlier-a
    X_train = X_train.drop(outliers_train)
    y_train = y_train.drop(outliers_train)
    X_test = X_test.drop(outliers_test)
    y_test = y_test.drop(outliers_test)

    X_train, X_test = standardize_data(X_train, X_test)

    return X_train, y_train, X_test, y_test



def main(train_file, test_file):
    train_data, test_data = load_data(train_file, test_file)

    X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)

    knn_model = CustomKNN(n_neighbors=5)
    knn_model.fit(X_train.values, y_train.values)

    y_pred = knn_model.predict(X_test.values)

    rmse = np.sqrt(np.mean((y_pred - y_test.values) ** 2))
    print(rmse)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python kod.py train_file.tsv test_file.tsv")
        sys.exit(1)
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    main(train_file, test_file)
