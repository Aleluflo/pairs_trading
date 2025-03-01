import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
import pandas as pd
import numpy as np

def get_adj_closes(tickers, start_date=None, end_date=None, freq='1d'):
    data = yf.download(tickers, start=start_date, end=end_date, interval=freq, auto_adjust=True)
    data.sort_index(inplace=True)
    return data['Close']

def adf_test(series):
    result = adfuller(series)
    return result[0], result[1]  # ADF Statistic, p-value

def check_cointegration(series1, series2):
    score, p_value, _ = coint(series1, series2)
    return score, p_value

class KalmanFilterHedgeRatio:
    def __init__(self):
        self.x = np.array([0, 0])
        self.A = np.eye(2)
        self.Q = np.eye(2) * 0.01
        self.R = np.array([[1]]) * 100
        self.P = np.eye(2) * 1000

    def predict(self):
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x, y):
        C = np.array([[1, x]])
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)
        self.P = (np.eye(2) - K @ C) @ self.P
        self.x = self.x + K @ (y - C @ self.x)
