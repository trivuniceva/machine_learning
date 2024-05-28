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


def preprocess_data(train_data, test_data):
    train_data = pd.get_dummies(data, columns=['Marka', 'Godina proizvodnje', 'Karoserija', 'Menjac'])
    test_data = pd.get_dummies(data, columns=['Marka', 'Godina proizvodnje', 'Karoserija', 'Menjac'])

    train_data['Menjac'] = train_data['Menjac'].map({'Manuelni': 0, 'Automatski': 1})
    test_data['Menjac'] = test_data['Menjac'].map({'Manuelni': 0, 'Automatski': 1})

    # Transformacija obeležja "Godina proizvodnje" u "Starost"
    train_data['Starost'] = 2023 - train_data['Godina proizvodnje']
    test_data['Starost'] = 2023 - test_data['Godina proizvodnje']

    # Izostavljanje obeležja "Gorivo" i "Grad"
    X_train = train_data[['Marka', 'Starost', 'Karoserija', 'Menjac']]
    y_train = train_data['Cena']
    X_test = test_data[['Marka', 'Starost', 'Karoserija', 'Menjac']]
    y_test = test_data['Cena']

    # Pretvaranje podataka u numerički format
    X_train = X_train.apply(pd.to_numeric, errors='coerce')
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    return X_train, y_train, X_test, y_test



def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def calculate_cramers_v(data, categorical_features):
    results = []
    for f1, f2 in combinations(categorical_features, 2):
        confusion_matrix = pd.crosstab(data[f1], data[f2])
        v = cramers_v(confusion_matrix.values)
        results.append((f1, f2, v))
    return results


def main(train_file, test_file):
    train_data, test_data = load_data(train_file, test_file)

    X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)

    knn_model = CustomKNN(n_neighbors=5)
    knn_model.fit(X_train.values, y_train.values)

    y_pred = knn_model.predict(X_test.values)

    rmse = np.sqrt(np.mean((y_pred - y_test.values) ** 2))
    print("RMSE:", rmse)

    # Compute and visualize correlation matrix
    # correlation_matrix = train_data[['Kilometraza', 'Konjske snage', 'Menjac', 'Cena']].corr()
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    # plt.title('Correlation Matrix')
    # plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python kod.py train_file.tsv test_file.tsv")
        sys.exit(1)
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    main(train_file, test_file)
