"Forecast future data points based on past data"

import json
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error


def make_lagged_supervised(series: List[float], n_lags: int):
    s = np.asarray(series, dtype=float)
    if len(s) <= n_lags:
        raise ValueError(f"Need more than n_lags={n_lags} points (got {len(s)}).")
    X, y = [], []
    for i in range(n_lags, len(s)):
        X.append(s[i - n_lags : i])
        y.append(s[i])
    return np.array(X), np.array(y)


@dataclass
class Forecaster:
    n_lags: int = 14
    model: RidgeCV = None

    def __post_init__(self):
        if self.model is None:
            self.model = RidgeCV(alphas=np.logspace(-3, 3, 13))

    def fit(self, series: List[float]) -> "Forecaster":
        X, y = make_lagged_supervised(series, self.n_lags)
        self.model.fit(X, y)
        return self

    def predict_next(self, history: List[float], horizon: int) -> List[float]:
        history = list(map(float, history))
        if len(history) < self.n_lags:
            raise ValueError(f"history length must be >= n_lags ({self.n_lags})")
        preds = []
        buffer = history[-self.n_lags :].copy()
        for _ in range(horizon):
            x = np.array(buffer)[-self.n_lags :][None, :]
            yhat = float(self.model.predict(x)[0])
            preds.append(yhat)
            buffer.append(yhat)
        return preds

    def backtest_holdout(self, series: List[float], holdout: int = 30):
        if holdout <= 0:
            raise ValueError("holdout must be >= 1")
        if len(series) <= self.n_lags + holdout:
            raise ValueError("Increase data length or reduce holdout/n_lags.")
        train = series[:-holdout]
        test = series[-(holdout + self.n_lags) :]

        self.fit(train)

        y_true, y_pred = [], []
        window = list(test[: self.n_lags])
        remaining = list(test[self.n_lags :])
        for actual in remaining:
            x = np.array(window)[-self.n_lags :][None, :]
            yhat = float(self.model.predict(x)[0])
            y_true.append(float(actual))
            y_pred.append(yhat)
            window.append(actual)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        return mae, rmse, np.array(y_true), np.array(y_pred)


def predict_next(process_func, n_lags: int, horizon: int) -> list:

    training_data = process_func()
    forecaster = Forecaster(n_lags=n_lags).fit(training_data)
    predicted = forecaster.predict_next(training_data, horizon=horizon)

    return predicted
