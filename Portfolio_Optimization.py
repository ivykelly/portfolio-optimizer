import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv()
import yfinance as yf
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize
from fredapi import Fred
fred = Fred(api_key = os.environ.get('FRED_API_KEY'))
import matplotlib.pyplot as plt

def main():
    tickers = ['SPY', 'BND', 'GLD', 'QQQ', 'VTI', 'AAPL'] # Change here if desired.

    os.makedirs('results', exist_ok=True)

    # Set timeframe to past 5 years
    end_date = datetime.today()
    start_date = end_date - timedelta(days = 5 * 365)

    # Create DataFrame to store the adjusted close prices. 
    adj_close_df = pd.DataFrame()

    for ticker in tickers:
        data = yf.download(ticker, start = start_date, end = end_date, auto_adjust=True)
        adj_close_df[ticker] = data["Close"]

    # Calculate logarithmic returns for each ticker
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns = log_returns.dropna()

    # Calculate covariance matrix using annualized log returns
    cov_matrix = log_returns.cov()*252 

    # Calculate the portfolio standard deviation
    def standard_deviation(weights, cov_matrix):
        variance = weights.T @ cov_matrix @ weights
        return np.sqrt(variance)

    # Calculate expected returns
    def expected_return(weights, log_returns):
        return np.sum(log_returns.mean()*weights)*252

    # Calculate Sharpe Ratio: (Expected Return - Risk Free Rate) / Standard Dev.
    def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

    # Calculate Max Drawdown
    def max_drawdown(cumulative_returns):
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()

    # Calculate Calmar Ratio
    def calmar_ratio(returns, cumulative_returns):
        ann_return = returns.mean() * 252
        mdd = abs(max_drawdown(cumulative_returns))
        
        if mdd == 0:
            return np.nan  # avoid division by zero
        
        return ann_return / mdd

    # Use the latest available 10-year Treasury yield as the risk-free rate
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
    risk_free_rate = ten_year_treasury_rate.iloc[-1] 

    # Negate Sharpe Ratio because scipy.optimize.minimize performs minimization
    def neg_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

    # Constrain weights to 0–50% and require weights to sum to 100%
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(0, 0.5) for _ in range(len(tickers))] 

    # Initialize optimization with equal portfolio weights
    initial_weights = np.array([1/len(tickers)]*len(tickers))

    # Optimize weights to maximize Sharpe Ratio
    optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns, cov_matrix, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)
    optimal_weights = optimized_results.x 

    # Display analytics of the optimal portfolio
    print("Optimal weights: ")
    for ticker, weight in zip(tickers, optimal_weights): 
        print(f"{ticker}: {weight: .4f}") 

    # Create a bar chart of the optimal weights
    plt.figure(figsize=(10,6))
    plt.bar(tickers, optimal_weights)

    plt.xlabel('Assets')
    plt.ylabel('Optimal Weights')
    plt.title('Optimal Portfolio Weights')

    plt.savefig('results/optimal_weights.png')
    plt.close()

    print()

    portfolio_returns = log_returns @ optimal_weights
    cumulative_returns = np.exp(portfolio_returns.cumsum())
    cumulative_returns /= cumulative_returns.iloc[0] # Normalize portfolio to start at exactly 1.0 

    # Calculate and display metrics
    optimal_portfolio_return = expected_return(optimal_weights, log_returns)
    optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
    optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)
    max_dd_sharpe = max_drawdown(cumulative_returns)
    calmar_sharpe = calmar_ratio(portfolio_returns, cumulative_returns)

    print(f"Expected Annual Return: {optimal_portfolio_return: .4f}")
    print(f"Expected Volatility: {optimal_portfolio_volatility: .4f}")
    print(f"Sharpe Ratio: {optimal_sharpe_ratio: .4f}")
    print(f"Max Drawdown: {max_dd_sharpe: .4f}")
    print(f"Calmar Ratio: {calmar_sharpe: .4f}")
    print()

    # Plot portfolio growth
    plt.figure(figsize=(10,6))
    plt.plot(cumulative_returns)
    plt.title("Optimal Portfolio Growth Over Time")
    plt.xlabel("Date")
    plt.ylabel("Growth")
    plt.savefig('results/optimal_portfolio_growth.png')
    plt.close()

    # Calculate optimal weights in terms of highest return only. 
    print("For contrast: if we base our portfolio optimization solely on returns, these will be the results.")
    print()

    # Make negative expected returns function. 
    def neg_expected_return(weights, log_returns):
        return -expected_return(weights, log_returns)

    # Optimize weights based on max returns 
    optimized_returns = minimize(neg_expected_return, initial_weights, args=(log_returns,), method='SLSQP', constraints=constraints, bounds=bounds)
    optimized_returns_weights = optimized_returns.x 

    print("Optimal weights in terms of highest returns only: ")
    for ticker, weight in zip(tickers, optimized_returns_weights):
        print(f"{ticker}: {weight: .4f}")

    print()

    # Calculate new cumulative returns
    optimized_returns_portfolio_returns = log_returns @ optimized_returns_weights 
    optimized_returns_cumulative_returns = np.exp(optimized_returns_portfolio_returns.cumsum())
    optimized_returns_cumulative_returns /= optimized_returns_cumulative_returns.iloc[0]

    # Calculate and display metrics
    optimal_returns_portfolio_return = expected_return(optimized_returns_weights, log_returns)
    optimal_returns_portfolio_volatility = standard_deviation(optimized_returns_weights, cov_matrix)
    optimal_returns_sharpe_ratio = sharpe_ratio(optimized_returns_weights, log_returns, cov_matrix, risk_free_rate)
    max_dd_return = max_drawdown(optimized_returns_cumulative_returns)
    calmar_returns = calmar_ratio(optimized_returns_portfolio_returns, optimized_returns_cumulative_returns)

    print(f"Expected Annual Return: {optimal_returns_portfolio_return: .4f}")
    print(f"Expected Volatility: {optimal_returns_portfolio_volatility: .4f}")
    print(f"Sharpe Ratio: {optimal_returns_sharpe_ratio: .4f}")
    print(f"Max Drawdown: {max_dd_return: .4f}")
    print(f"Calmar Ratio: {calmar_returns: .4f}")
    print()

    print("We can also see the results if basing the optimization solely on minimum volatility")

    print()

    # Calculate optimal weights in terms of lowest volatility only
    optimized_volatility = minimize(standard_deviation, initial_weights, args=(cov_matrix,), method='SLSQP', constraints=constraints, bounds=bounds)
    optimized_volatility_weights = optimized_volatility.x 

    print("Optimal weights in terms of lowest volatility only: ")
    for ticker, weight in zip(tickers, optimized_volatility_weights):
        print(f"{ticker}: {weight: .4f}")

    print()

    optimized_volatility_portfolio_returns = log_returns @ optimized_volatility_weights
    optimized_volatility_cumulative_returns = np.exp(optimized_volatility_portfolio_returns.cumsum())
    optimized_volatility_cumulative_returns /= optimized_volatility_cumulative_returns.iloc[0]

    # Calculate and display metrics
    optimal_volatility_portfolio_return = expected_return(optimized_volatility_weights, log_returns)
    optimal_volatility_portfolio_volatility = standard_deviation(optimized_volatility_weights, cov_matrix)
    optimal_volatility_sharpe_ratio = sharpe_ratio(optimized_volatility_weights, log_returns, cov_matrix, risk_free_rate)
    max_dd_vol = max_drawdown(optimized_volatility_cumulative_returns)
    calmar_volatility = calmar_ratio(optimized_volatility_portfolio_returns, optimized_volatility_cumulative_returns)

    print(f"Expected Annual Return: {optimal_volatility_portfolio_return: .4f}")
    print(f"Expected Volatility: {optimal_volatility_portfolio_volatility: .4f}")
    print(f"Sharpe Ratio: {optimal_volatility_sharpe_ratio: .4f}")
    print(f"Max Drawdown: {max_dd_vol: .4f}")
    print(f"Calmar Ratio: {calmar_volatility: .4f}")
    print()

    # Compare portfolio growth for each case on one plot
    plt.figure(figsize=(10,6))
    plt.plot(cumulative_returns, label="Max Sharpe Ratio")
    plt.plot(optimized_returns_cumulative_returns, label="Max Returns")
    plt.plot(optimized_volatility_cumulative_returns, label="Min Volatility")
    plt.title("Portfolio Growth Under Different Optimization Objectives")
    plt.xlabel("Date")
    plt.ylabel("Growth")
    plt.legend()
    plt.grid(True)
    plt.savefig('results/portfolio_growth_comparison.png')
    plt.close()


    # Print a table comparing weights
    weights_df = pd.DataFrame({
        "Ticker": tickers,
        "Max Sharpe": optimal_weights,
        "Max Return": optimized_returns_weights,
        "Min Volatility": optimized_volatility_weights
    })

    weights_df = weights_df.set_index("Ticker")
    weights_df.round(4).to_csv('results/weights_comparison.csv')

    print("Portfolio Weights Comparison:")
    print(weights_df.round(4))

    print()


    # Compare optimized portfolio against market benchmarks, the risk-free rate, and a 60/40 portfolio

    # Download data for ETFs
    benchmark_tickers = ['SPY', 'VTI', 'QQQ', 'AGG']

    benchmark_closes_df = pd.DataFrame()

    for ticker in benchmark_tickers:
        data = yf.download(ticker, start = start_date, end = end_date, auto_adjust=True)
        benchmark_closes_df[ticker] = data["Close"]

    log_returns_benchmarks = np.log(benchmark_closes_df / benchmark_closes_df.shift(1))
    log_returns_benchmarks = log_returns_benchmarks.dropna()

    cumulative_returns_benchmarks = np.exp(log_returns_benchmarks.cumsum())
    cumulative_returns_benchmarks /= cumulative_returns_benchmarks.iloc[0] 


    # Now for risk-free-rate: we've already drawn it from real financial data. Let's convert it into daily return. 
    daily_rf = np.log(1 + risk_free_rate) / 252

    # Create time series matching portfolio dates. 
    rf_returns = pd.Series(daily_rf, index=log_returns.index)
    rf_cumulative = np.exp(rf_returns.cumsum())
    rf_cumulative /= rf_cumulative.iloc[0]


    # Finally, let's do the 60/40 portfolio. 
    weights_6040 = np.array([0.6, 0.4])
    returns_6040 = log_returns_benchmarks[['SPY', 'AGG']] @ weights_6040
    cumulative_6040 = np.exp(returns_6040.cumsum())
    cumulative_6040 /= cumulative_6040.iloc[0]

    # Plot growth of each one compared to optimized Sharpe Ratio Portfolio
    plt.figure(figsize=(10,6))
    
    for ticker in cumulative_returns_benchmarks.columns:
        plt.plot(cumulative_returns_benchmarks[ticker], label=ticker)

    plt.plot(cumulative_returns, label="Optimized")

    plt.plot(rf_cumulative, label="Risk-Free (Savings Account)")

    plt.plot(cumulative_6040, label="60/40 Portfolio")

    plt.title("Optimized Portfolio vs Market Benchmarks")
    plt.xlabel("Date")
    plt.ylabel("Growth")
    plt.legend()
    plt.grid(True)
    plt.savefig('results/optimized_vs_benchmarks_growth.png')
    plt.close()


    # Compare metrics for all strategies. 

    # First, calculate return, volatility, Sharpe Ratio, Max Drawdown, and Calmar Ratio of each ETF
    results = []
    for ticker in benchmark_tickers:
        weights = np.array([1.0])  # 100% in one asset
        
        returns = log_returns_benchmarks[[ticker]]
        cov = returns.cov() * 252 
        
        r = expected_return(weights, returns)
        vol = standard_deviation(weights, cov)
        sharpe = sharpe_ratio(weights, returns, cov, risk_free_rate)
        max_dd = max_drawdown(cumulative_returns_benchmarks[ticker])
        
        benchmark_returns = log_returns_benchmarks[ticker]
        calmar = calmar_ratio(benchmark_returns, cumulative_returns_benchmarks[ticker])

        results.append([ticker, r, vol, sharpe, max_dd, calmar])

    # Compute 60/40 separately
    r_6040 = returns_6040.mean() * 252
    vol_6040 = returns_6040.std() * np.sqrt(252)
    sharpe_6040 = (r_6040 - risk_free_rate) / vol_6040
    max_dd_6040 = max_drawdown(cumulative_6040)
    calmar_6040 = calmar_ratio(returns_6040, cumulative_6040)

    results.append(["60/40", r_6040, vol_6040, sharpe_6040, max_dd_6040, calmar_6040])

    # Omit risk-free metrics: with a constant risk-free rate, volatility and max drawdown
    # are zero, making Sharpe and Calmar ratios undefined. 

    # Add optimized portfolio results
    optimized_results = [
        ["Max Sharpe", optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio, max_dd_sharpe, calmar_sharpe],
        ["Max Return", optimal_returns_portfolio_return, optimal_returns_portfolio_volatility, optimal_returns_sharpe_ratio, max_dd_return, calmar_returns],
        ["Min Volatility", optimal_volatility_portfolio_return, optimal_volatility_portfolio_volatility, optimal_volatility_sharpe_ratio, max_dd_vol, calmar_volatility]
    ]

    all_results = optimized_results + results

    print("Let's compare returns, volatility, and Sharpe Ratios of all strategies.")
    print()

    comparison_df = pd.DataFrame(
        all_results,
        columns=["Portfolio", "Return", "Volatility", "Sharpe Ratio", "Max Drawdown", "Calmar Ratio"]
    )

    comparison_df = comparison_df.set_index("Portfolio")
    comparison_df.round(4).to_csv('results/strategy_comparison.csv')

    print(comparison_df.round(4))

if __name__ == '__main__': main() 