import sys
import numpy as np
import pandas as pd
import math


class LinearRegression:
    def __init__(self, degree=5):
        self.coefficients = None
        self.degree = degree

    def normalize_data(self, data):
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1, range_vals)
        normalized_data = (data - min_vals) / range_vals
        return normalized_data

    def denormalize_data(self, normalized_data, original_data):
        denormalized_data = normalized_data * (np.max(original_data, axis=0) - np.min(original_data, axis=0)) + np.min(
            original_data, axis=0)
        return denormalized_data

    def fit_normal_equation(self, x, y):
        x = np.column_stack((np.ones(len(x)), x))
        self.coefficients = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

        return self.coefficients

    def fit_gradient_descent_batch(self, x, y, learning_rate, epochs):
        x = np.column_stack((np.ones(len(x)), x))
        self.coefficients = np.zeros(x.shape[1])

        for epoch in range(epochs):
            predictions = np.dot(x, self.coefficients)
            errors = predictions - y
            gradient = np.dot(errors, x) / len(y)

            self.coefficients -= learning_rate * gradient

        return self.coefficients

    def fit_gradient_descent_stohastic(self, x, y, learning_rate, epochs):
        x = np.column_stack((np.ones(len(x)), x))
        self.coefficients = np.zeros(x.shape[1])

        for epoch in range(epochs):
            for i in range(len(y)):
                rand_idx = np.random.randint(len(y))
                x_i = x[rand_idx:rand_idx + 1]
                y_i = y[rand_idx:rand_idx + 1]

                prediction = np.dot(x_i, self.coefficients)
                error = prediction - y_i

                gradient = np.dot(error, x_i)
                self.coefficients -= learning_rate * gradient

        return self.coefficients

    def fit_poly_sec(self, x, y):
        X_poly = np.column_stack([x ** i for i in range(1, self.degree + 1)])
        X_poly = np.column_stack((np.ones(len(x)), X_poly))
        self.coefficients = np.linalg.inv(X_poly.T.dot(X_poly)).dot(X_poly.T).dot(y)

        return self.coefficients

    def fit_poly_fifth(self, x, y, lambda_reg=0.01):
        X_poly = np.column_stack([x ** i for i in range(1, self.degree + 1)])
        X_poly = np.column_stack((np.ones(len(x)), X_poly, x ** 3, x ** 4, x ** 5))
        regularization_term = lambda_reg * np.eye(X_poly.shape[1])
        self.coefficients = np.linalg.inv(X_poly.T.dot(X_poly) + regularization_term).dot(X_poly.T).dot(y)
        return self.coefficients

    def predict_poly(self, x):
        X_poly = np.column_stack([x ** i for i in range(1, self.degree + 1)])
        X_poly = np.column_stack((np.ones(len(x)), X_poly, x ** 3, x ** 4, x ** 5))
        predictions = np.dot(X_poly, self.coefficients)

        return predictions

    def predict(self, x):
        x = np.column_stack((np.ones(len(x)), x))
        predictions = np.dot(x, self.coefficients)

        return predictions

    def calculate_rmse(self, y_true, y_predict):
        rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(y_predict, y_true)) / len(y_true))
        return rmse


def train_validate_test_split(data, train_percent=0.7, validate_percent=0.15, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(len(data))
    m = len(data)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train_data = data.iloc[perm[:train_end]]
    validate_data = data.iloc[perm[train_end:validate_end]]
    test_data = data.iloc[perm[validate_end:]]
    return train_data, validate_data, test_data


def calculate_zscore(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = (data - mean) / std_dev
    return z_scores


def detect_outliers(data, threshold=3):
    z_scores = calculate_zscore(data)
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
    return outlier_indices


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#
#
#
#
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

train_path = sys.argv[1]
test_path = sys.argv[2]

data = pd.read_csv(train_path)
train_data, validate_data, test_data = train_validate_test_split(data)

test_data_final = pd.read_csv(test_path)

X_train = train_data['X'].values
Y_train = train_data['Y'].values
X_validate = validate_data['X'].values
Y_validate = validate_data['Y'].values
X_test = test_data['X'].values
Y_test = test_data['Y'].values

X_test_final = test_data_final['X'].values.reshape(-1, 1)
Y_test_final = test_data_final['Y'].values

X_train = X_train.reshape(-1, 1)
X_validate = X_validate.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

outlier_indices = detect_outliers(Y_train)

threshold = 3
outlier_indices = detect_outliers(Y_train, threshold)

X_train_cleaned = np.delete(X_train, outlier_indices, axis=0)
Y_train_cleaned = np.delete(Y_train, outlier_indices, axis=0)


# learning_rate = 0.00025
learning_rate = 0.00025
epochs = 1000

model = LinearRegression()

# coefficients = model.fit_gradient_descent_batch(X_train_cleaned, Y_train_cleaned, learning_rate, epochs)
# coefficients = model.fit_gradient_descent_stohastic(X_train_cleaned, Y_train_cleaned, learning_rate, epochs)
# coefficients = model.fit_normal_equation(X_train_cleaned, Y_train_cleaned)
coefficients = model.fit_poly_fifth(X_train_cleaned, Y_train_cleaned, lambda_reg=0.01)

# predictions_validate = model.predict(X_test)
predictions_validate = model.predict_poly(X_test)

# Računanje RMSE na validacionom skupu
rmse_validate = model.calculate_rmse(Y_test, predictions_validate)
# print(f"RMSE on Validation Set: {rmse_validate}")

# predictions_test = model.predict(X_test_final)
predictions_test = model.predict_poly(X_test_final)

# Računanje RMSE na test skupu
rmse_test = model.calculate_rmse(Y_test_final, predictions_test)
print(rmse_test)
