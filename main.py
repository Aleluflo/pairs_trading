from utils import get_adj_closes, adf_test, check_cointegration, KalmanFilterHedgeRatio
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pandas as pd
import numpy as np
import ta


def main():
    tickers = ['GOOGL', 'VOO']
    closes = get_adj_closes(tickers, start_date='2015-01-01', end_date='2025-01-01')

    results_adf = {}
    results_coint = {}

    for i in range(len(tickers)):
        adf_stat, adf_p_value = adf_test(closes[tickers[i]])
        results_adf[tickers[i]] = (adf_stat, adf_p_value)

        for j in range(i + 1, len(tickers)):
            coint_score, coint_p_value = check_cointegration(closes[tickers[i]], closes[tickers[j]])
            results_coint[(tickers[i], tickers[j])] = (coint_score, coint_p_value)

    print("Unit Root Tests: (Augmented) Dickey-Fuller")
    for ticker, (adf_stat, adf_p_value) in results_adf.items():
        print(f'{ticker}: ADF Statistic = {adf_stat}, p-value = {adf_p_value}')

    print("\nEngle-Granger Two-Step Method: Cointegration")
    for pair, (coint_score, coint_p_value) in results_coint.items():
        print(f'Cointegration test for {pair}: score = {coint_score}, p-value = {coint_p_value}')

    closes.plot(figsize=(14, 7))
    plt.title('Adjusted Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Closing Price')
    plt.legend(closes.columns)
    plt.grid(True)
    plt.show()

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

    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    print(model.summary())

    data = closes[['GOOGL', 'VOO']].values
    johansen_test = coint_johansen(data, det_order=0, k_ar_diff=1)

    cointegration_vector = johansen_test.evec[:, 0]

    spread = data @ cointegration_vector

    spread_mean = np.mean(spread)
    spread_std = np.std(spread)
    spread_normalized = (spread - spread_mean) / spread_std

    spread_model = f"u_t = {cointegration_vector[0]:.5f} * x_t - {cointegration_vector[1]:.5f} * y_t"

    plt.figure(figsize=(12, 6))
    plt.plot(closes.index, spread_normalized, label="spread normalized", color="black")

    sigma_levels = {1.5: "brown", 2: "orange", 2.5: "green"}
    for sigma, color in sigma_levels.items():
        plt.axhline(sigma, color=color, linestyle='-', linewidth=1.5, label=f"{sigma} sigma")
        plt.axhline(-sigma, color=color, linestyle='-', linewidth=1.5)

    plt.axhline(0, color='red', linestyle='--', linewidth=2, label="Mean (0)")

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
        x_ = closes['VOO'].iloc[i]
        y_ = closes['GOOGL'].iloc[i]
        kalman_filter.update(x_, y_)

        w1 = kalman_filter.x[1]
        w1 = np.clip(w1, -10, 10)
        hedge_ratios.append(w1)
        spread = y_ - w1 * x_
        spread_values.append(spread)

    closes['HedgeRatio'] = hedge_ratios
    closes['Spread'] = spread_values

    window = 60
    closes['SpreadMean'] = closes['Spread'].rolling(window).mean()
    closes['SpreadStd'] = closes['Spread'].rolling(window).std()
    closes['ZScore'] = (closes['Spread'] - closes['SpreadMean']) / closes['SpreadStd']

    closes['RSI'] = ta.momentum.RSIIndicator(closes['GOOGL']).rsi()

    closes['LongSignal'] = (closes['ZScore'] < -2.5) & (closes['RSI'] < 40)
    closes['ShortSignal'] = (closes['ZScore'] > 2.5) & (closes['RSI'] > 60)

    initial_capital = 1_000_000
    capital = initial_capital
    position = 0
    last_trade_date = None
    min_days_between_trades = 5

    entry_price_googl = 0
    entry_price_voo = 0
    returns = [capital]

    for i in range(1, len(closes)):
        row = closes.iloc[i]
        prev_position = position

        price_googl = row['GOOGL']
        price_voo = row['VOO']
        hedge_ratio = row['HedgeRatio']

        z_abs = abs(row['ZScore'])
        position_size = capital * (0.05 + 0.05 * min(z_abs, 3))
        n_shares = position_size // (price_googl + hedge_ratio * price_voo)

        if prev_position == 1 and row['ZScore'] > -0.5:
            pnl = (price_googl - entry_price_googl) - hedge_ratio * (price_voo - entry_price_voo)
            capital += pnl * n_shares
            position = 0

        elif prev_position == -1 and row['ZScore'] < 0.5:
            pnl = -(price_googl - entry_price_googl) + hedge_ratio * (price_voo - entry_price_voo)
            capital += pnl * n_shares
            position = 0

        if position == 0:
            if row['LongSignal'] and (
                    last_trade_date is None or (row.name - last_trade_date).days >= min_days_between_trades):
                entry_price_googl = price_googl
                entry_price_voo = price_voo
                position = 1
                last_trade_date = row.name

            elif row['ShortSignal'] and (
                    last_trade_date is None or (row.name - last_trade_date).days >= min_days_between_trades):
                entry_price_googl = price_googl
                entry_price_voo = price_voo
                position = -1
                last_trade_date = row.name

        returns.append(capital)

    returns_df = pd.DataFrame({'Capital': returns}, index=closes.index)

    plt.figure(figsize=(12, 5))
    plt.plot(returns_df.index, returns_df['Capital'], label='Capital', color='blue')
    plt.axhline(initial_capital, color='red', linestyle='--', label='Initial Capital')
    plt.legend()
    plt.title('Capital Growth - Pairs Trading Strategy')
    plt.show()

    final_capital = returns_df['Capital'].iloc[-1]
    percentage_return = (final_capital - initial_capital) / initial_capital * 100

    results = {
        'Initial Capital': initial_capital,
        'Final Capital': final_capital,
        'Total Return (%)': percentage_return
    }

    results_df = pd.DataFrame([results])
    print(results_df)


if __name__ == "__main__":
    main()
