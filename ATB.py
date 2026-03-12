#2019 At Leibniz Institute of Agricultural Engineering and Bio-economy e.V. (ATB)

# Polynomial Regression for Pressure Drop in Porous Media
# ΔP/l = (μD + 1/2 Fρ|u|)u  → polynomial regression of degree 2

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


class DataLoader:
    """Handles loading and preparing data."""

    def __init__(self):
        self.x = None
        self.y = None

    def load_data(self):
        self.x = np.array([0,0.1,0.2,0.5,0.8,1.2,1.5]).reshape(-1,1)
        self.y = np.array([0,0.00073456,0.00282897,0.0176148,
                           0.0448887,0.0985493,0.151895]).reshape(-1,1)
        return self.x, self.y


class PolynomialRegressionModel:
    """Builds and trains the polynomial regression model."""

    def __init__(self, degree=2):
        self.degree = degree
        self.transformer = PolynomialFeatures(degree=self.degree, include_bias=False)
        self.model = LinearRegression()
        self.x_poly = None

    def transform_features(self, x):
        self.x_poly = self.transformer.fit_transform(x)
        return self.x_poly

    def train(self, x, y):
        self.transform_features(x)
        self.model.fit(self.x_poly, y)

    def predict(self, x):
        x_poly = self.transformer.transform(x)
        return self.model.predict(x_poly)

    def get_coefficients(self):
        return self.model.intercept_, self.model.coef_

    def score(self, x, y):
        x_poly = self.transformer.transform(x)
        return self.model.score(x_poly, y)


class ModelEvaluator:
    """Evaluates model performance."""

    @staticmethod
    def compute_rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))


class Visualizer:
    """Handles plotting of regression results."""

    @staticmethod
    def plot_results(x, y, y_pred):
        plt.figure()

        plt.scatter(x, y, label="Data points")
        plt.plot(x, y_pred, color='red', label="Regression curve")

        plt.xlabel("Velocity (u)")
        plt.ylabel("Pressure drop")
        plt.title("Polynomial Regression Fit")

        plt.legend()
        plt.savefig("polynomial_regression_fit.png", dpi=300, bbox_inches='tight')
        plt.show()


class RegressionPipeline:
    """Full workflow pipeline."""

    def __init__(self):
        self.data_loader = DataLoader()
        self.model = PolynomialRegressionModel(degree=2)
        self.evaluator = ModelEvaluator()
        self.visualizer = Visualizer()

    def run(self):
        # Load data
        x, y = self.data_loader.load_data()

        # Train model
        self.model.train(x, y)

        # Predictions
        y_pred = self.model.predict(x)

        # Metrics
        r2 = self.model.score(x, y)
        rmse = self.evaluator.compute_rmse(y, y_pred)

        # Coefficients
        intercept, coef = self.model.get_coefficients()

        print("R² score:", r2)
        print("Intercept (b0):", intercept)
        print("Coefficients (b1, b2):", coef)
        print("RMSE:", rmse)

        # Plot
        self.visualizer.plot_results(x, y, y_pred)


if __name__ == "__main__":
    pipeline = RegressionPipeline()
    pipeline.run()

