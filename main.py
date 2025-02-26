from utils import get_adj_closes, adf_test, check_cointegration
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pandas as pd
import numpy as np

def main():
    tickers = ['GOOGL', 'VOO']
    closes = get_adj_closes(tickers, start_date='2015-01-01', end_date='2025-01-01')

    results_adf = {}
    results_coint = {}

    # Verificar raíz unitaria y cointegración
    for i in range(len(tickers)):
        adf_stat, adf_p_value = adf_test(closes[tickers[i]])
        results_adf[tickers[i]] = (adf_stat, adf_p_value)

        for j in range(i + 1, len(tickers)):
            coint_score, coint_p_value = check_cointegration(closes[tickers[i]], closes[tickers[j]])
            results_coint[(tickers[i], tickers[j])] = (coint_score, coint_p_value)

    # Mostrar resultados
    print("Unit Root Tests: (Augmented) Dickey-Fuller")
    for ticker, (adf_stat, adf_p_value) in results_adf.items():
        print(f'{ticker}: ADF Statistic = {adf_stat}, p-value = {adf_p_value}')

    print("\nEngle-Granger Two-Step Method: Cointegration")
    for pair, (coint_score, coint_p_value) in results_coint.items():
        print(f'Cointegration test for {pair}: score = {coint_score}, p-value = {coint_p_value}')

    # Graficar los precios
    closes.plot(figsize=(14, 7))
    plt.title('Precios de Cierre Ajustados')
    plt.xlabel('Fecha')
    plt.ylabel('Precio de Cierre Ajustado')
    plt.legend(closes.columns)
    plt.grid(True)
    plt.show()

    # Modelo de regresión y prueba de Johansen
    closes.dropna(inplace=True)
    Y = closes['GOOGL']
    X = closes['VOO']

    spread = Y - (X * 0.4025 - 33.4484)
    adf_stat, p_value = adfuller(spread)[:2]
    print(f"\nADF Test Statistic: {adf_stat}")
    print(f"P-value: {p_value}")
    if p_value < 0.05:
        print("The spread is stationary (cointegration detected).")
    else:
        print("The spread is NOT stationary (no cointegration).")

    # Regresión OLS
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print(model.summary())

    # Prueba de Johansen
    data = closes[['GOOGL', 'VOO']].values
    johansen_test = coint_johansen(data, det_order=0, k_ar_diff=1)
    cointegration_vector = johansen_test.evec[:, 0]

    spread = data @ cointegration_vector
    spread_normalized = (spread - np.mean(spread)) / np.std(spread)
    spread_model = f"u_t = {cointegration_vector[0]:.5f} * x_t - {cointegration_vector[1]:.5f} * y_t"

    print("Eigenvalues:\n", johansen_test.eig)
    print("Trace Statistics:\n", johansen_test.lr1)
    print("Eigenvectors:\n", johansen_test.evec)
    print("\nNormalized Spread Model:", spread_model)

    # Gráfico del spread normalizado
    plt.figure(figsize=(12, 6))
    plt.plot(spread_normalized, label="Normalized Spread", color="blue")
    plt.axhline(0, color='red', linestyle='--', label="Mean (0)")
    plt.legend()
    plt.title("Normalized Cointegrated Spread")
    plt.show()

if __name__ == "__main__":
    main()
