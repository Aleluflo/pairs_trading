from utils import get_adj_closes, adf_test, check_cointegration, KalmanFilterHedgeRatio
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pandas as pd
import numpy as np
import ta


def main():
    # Download prices
    tickers_test = ['GOOGL', 'IBM', 'AXP', 'AAPL', 'VOO', 'WMT', 'COST']
    closes_test = get_adj_closes(tickers_test, start_date='2015-01-01', end_date='2025-01-01')

    # Verificar raíz unitaria y cointegración entre pares de activos
    results_adf = {}
    results_coint = {}
    for i in range(len(tickers_test)):
        adf_stat, adf_p_value = adf_test(closes_test[tickers_test[i]])
        results_adf[tickers_test[i]] = (adf_stat, adf_p_value)

        for j in range(i + 1, len(tickers_test)):
            coint_score, coint_p_value = check_cointegration(closes_test[tickers_test[i]], closes_test[tickers_test[j]])
            results_coint[(tickers_test[i], tickers_test[j])] = (coint_score, coint_p_value)

    # Show uniroot results (Dickey-Fuller)
    print("Unit Root Tests: (Augmented) Dickey-Fuller")
    for ticker_test, (adf_stat, adf_p_value) in results_adf.items():
        print(f'{ticker_test}: ADF Statistic = {adf_stat}, p-value = {adf_p_value}')

    # Show cointegration results (Engle-Granger)
    print("\nEngle-Granger Two-Step Method: P1 linear relation")
    for pair, (coint_score, coint_p_value) in results_coint.items():
        print(f'Cointegration test for {pair}: score = {coint_score}, p-value = {coint_p_value}')

    # Plot the assets
    closes_test.plot(figsize=(14, 7))
    plt.title('Adjusted Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Closing Price')
    plt.legend(closes_test.columns)
    plt.grid(True)
    plt.show()

    normalized_closes = closes_test / closes_test.iloc[0] * 100

    plt.figure(figsize=(12, 6))
    for ticker_test in tickers_test:
        plt.plot(normalized_closes.index, normalized_closes[ticker_test], label=ticker_test)

    plt.title("Normalized Prices (Base 100)")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Download prices
    tickers = ['AXP', 'GOOGL']
    closes = get_adj_closes(tickers, start_date='2015-01-01', end_date='2025-01-01')

    # Drop missing values
    closes.dropna(inplace=True)

    # Define dependent and independent variables
    Y = closes['AXP']
    X = closes['GOOGL']

    # Test if the spread is stationary
    spread = Y - (X * 1.2073 + 21.2193)
    adf_result = adfuller(spread)
    adf_stat, p_value = adf_result[0], adf_result[1]

    print(f"ADF Test Statistic: {adf_stat}")
    print(f"P-value: {p_value}")
    if p_value < 0.05:
        print("The spread is stationary (cointegration detected).")
    else:
        print("The spread is NOT stationary (no cointegration).")

    # Run OLS regression
    X = sm.add_constant(X)  # Add constant term
    model = sm.OLS(Y, X).fit()

    # Print regression summary
    print(model.summary())

    data = closes[['AXP', 'GOOGL']].values
    # Run Johansen cointegration test
    johansen_test = coint_johansen(data, det_order=0, k_ar_diff=1)

    # Extract cointegration vector (eigenvector)
    cointegration_vector = johansen_test.evec[:, 0]

    # Compute the raw spread
    spread = data @ cointegration_vector  # Matrix multiplication of prices with eigenvector

    # Normalize the spread to have mean 0 and std 1
    spread_mean = np.mean(spread)
    spread_std = np.std(spread)
    spread_normalized = (spread - spread_mean) / spread_std

    # Construct the normalized spread model equation
    spread_model = f"u_t = {cointegration_vector[0]:.5f} * x_t - {cointegration_vector[1]:.5f} * y_t"

    # Plot the normalized spread with sigma levels
    plt.figure(figsize=(12, 6))
    plt.plot(closes.index, spread_normalized, label="spread normalized", color="black")

    # Sigma levels and their colors
    sigma_levels = {1.5: "brown", 2: "orange", 2.5: "green"}
    for sigma, color in sigma_levels.items():
        plt.axhline(sigma, color=color, linestyle='-', linewidth=1.5, label=f"{sigma} sigma")
        plt.axhline(-sigma, color=color, linestyle='-', linewidth=1.5)

    # Mean line
    plt.axhline(0, color='red', linestyle='--', linewidth=2, label="Mean (0)")

    # Formatting
    plt.title("VECM Signal with Sigma Levels", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Normalized Spread", fontsize=12)
    plt.legend(loc="upper left")
    plt.show()

    kalman_filter = KalmanFilterHedgeRatio()
    hedge_ratios = []
    spread_values = []

    for i in range(len(closes)):
        kalman_filter.predict()
        x_ = closes['GOOGL'].iloc[i]
        y_ = closes['AXP'].iloc[i]
        kalman_filter.update(x_, y_)

        w1 = kalman_filter.x[1]
        w1 = np.clip(w1, -10, 10)
        hedge_ratios.append(w1)
        spread = y_ - w1 * x_
        spread_values.append(spread)

    closes['HedgeRatio'] = hedge_ratios
    closes['Spread'] = spread_normalized

    # Gráfica de Hedge Ratio
    plt.figure(figsize=(12, 6))
    plt.plot(closes.index, closes['HedgeRatio'], label='Hedge Ratio', color='teal')
    plt.title('Evolución del Hedge Ratio (Filtro de Kalman)')
    plt.xlabel('Fecha')
    plt.ylabel('Hedge Ratio')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Gráfica de Spread
    plt.figure(figsize=(12, 6))
    plt.plot(closes.index, closes['Spread'], label='Spread', color='darkorange')
    plt.title('Evolución del Spread')
    plt.xlabel('Fecha')
    plt.ylabel('Spread')
    plt.legend()
    plt.grid(True)
    plt.show()

    shares = 100
    com = 0.125 / 100

    active_positions = []
    active_short_positions = []

    capital = 1_000_000
    portfolio_value = [capital]

    for i, row in closes.iterrows():
        spread = row['Spread']

        # Compra AXP y vende corto COST si spread > 1.5
        if spread < -1.5:
            if capital >= (row['AXP'] * shares * (1 + com)):
                capital -= row['AXP'] * shares * (1 + com)
                active_positions.append({
                    "date": row.name,
                    "ticker": 'AXP',
                    "bought_at": row['AXP'],
                    "shares": shares
                })
            if capital >= (row['GOOGL'] * shares * (1 + com)):
                capital -= row['GOOGL'] * shares * (com)
                active_short_positions.append({
                    "date": row.name,
                    "ticker": 'GOOGL',
                    "bought_at": row['GOOGL'],
                    "shares": shares
                })

        # Compra COST y vende corto AXP si spread < -1.5
        elif spread > 1.5:
            if capital >= (row['GOOGL'] * shares * (1 + com)):
                capital -= row['GOOGL'] * shares * (1 + com)
                active_positions.append({
                    "date": row.name,
                    "ticker": 'GOOGL',
                    "bought_at": row['GOOGL'],
                    "shares": shares
                })
            if capital >= (row['AXP'] * shares * com):
                capital -= row['AXP'] * shares * (com)
                active_short_positions.append({
                    "date": row.name,
                    "ticker": 'AXP',
                    "bought_at": row['AXP'],
                    "shares": shares
                })

        # Cierre de posiciones si spread es 0
        if abs(spread) <= 0.05:
            for pos in active_positions:
                capital += row[pos['ticker']] * pos['shares'] * (1 - com)
            active_positions.clear()
            for pos in active_short_positions:
                capital += (pos['bought_at'] - row[pos['ticker']]) * pos['shares'] - (row[pos['ticker']] * shares * com)
            active_short_positions.clear()

        active_val = sum([pos['shares'] * row[pos['ticker']] for pos in active_positions])
        short_val = sum([(pos['bought_at'] - row[pos['ticker']]) * pos['shares'] for pos in active_short_positions])

        current_value = active_val + short_val + capital
        portfolio_value.append(current_value)

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_value, label='Valor del portafolio', color='blue')
    plt.xlabel('Período')
    plt.ylabel('Valor del portafolio')
    plt.title('Evolución del valor del portafolio')
    plt.legend()
    plt.grid()
    plt.show()

    valor_final = portfolio_value[-1]
    print(f"Valor final del portafolio: ${valor_final:,.2f}")


if __name__ == "__main__":
    main()

