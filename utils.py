import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
import pandas as pd

def get_adj_closes(tickers, start_date=None, end_date=None, freq='1d'):
    """Descarga precios de cierre ajustados usando yfinance."""
    data = yf.download(tickers, start=start_date, end=end_date, interval=freq, auto_adjust=True)
    data.sort_index(inplace=True)
    return data['Close']

def adf_test(series):
    """Realiza la prueba de raíz unitaria Dickey-Fuller."""
    result = adfuller(series)
    return result[0], result[1]  # ADF Statistic, p-value

def check_cointegration(series1, series2):
    """Realiza la prueba de cointegración Engle-Granger."""
    score, p_value, _ = coint(series1, series2)
    return score, p_value
