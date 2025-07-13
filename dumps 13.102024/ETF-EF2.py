import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ETFDataHandler:
    def __init__(self, etfs, start_date, end_date):
        self.etfs = etfs
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.download_data()

    def download_data(self):
        data = yf.download(self.etfs, start=self.start_date, end=self.end_date)['Adj Close']
        print("Data downloaded successfully.")
        return data

class ETFMetrics:
    def __init__(self, etf_data_handler):
        self.etf_data = etf_data_handler.data
        self.daily_returns = self.calculate_daily_returns()
        self.metrics = self.calculate_individual_metrics()

    def calculate_daily_returns(self):
        daily_returns = self.etf_data.pct_change().dropna()
        return daily_returns

    def calculate_individual_metrics(self):
        metrics = pd.DataFrame(index=self.etf_data.columns)
        metrics['Max Return'] = self.daily_returns.max()
        metrics['Min Return'] = self.daily_returns.min()
        metrics['Standard Deviation'] = self.daily_returns.std() * np.sqrt(252)  # Annualize standard deviation
        metrics['Total Return'] = (1 + self.daily_returns).cumprod().iloc[-1] - 1
        metrics['Annualized Return'] = (1 + metrics['Total Return']) ** (1 / (len(self.daily_returns) / 252)) - 1
        metrics['Sharpe Ratio'] = metrics['Annualized Return'] / metrics['Standard Deviation']
        return metrics

class PortfolioOptimizer:
    def __init__(self, etf_metrics):
        self.daily_returns = etf_metrics.daily_returns
        self.num_assets = len(self.daily_returns.columns)
        self.results_matrix, self.optimal_weights, self.optimal_metrics = self.optimize_portfolio(10000)

    def generate_random_weights(self, num_assets):
        while True:
            max_etf_weight = 0.4
            # weights = np.random.choice(np.arange(0, 1.01, 0.01), size=num_assets)
            weights = np.random.choice(np.arange(0, 101, 1), size=num_assets) / 100
            weights = weights / np.sum(weights)
            if np.all(weights <= max_etf_weight):
                return weights

    def calculate_portfolio_performance(self, weights):
        weighted_returns = np.dot(self.daily_returns, weights)
        portfolio_return = (1 + weighted_returns).cumprod()[-1] - 1
        portfolio_std_dev = np.std(weighted_returns) * np.sqrt(252)  # Annualize standard deviation
        sharpe_ratio = portfolio_return / portfolio_std_dev
        return portfolio_return, portfolio_std_dev, sharpe_ratio

    def optimize_portfolio(self, num_portfolios):
        results_matrix = np.zeros((4, num_portfolios))
        optimal_weights = None
        max_sharpe_ratio = -np.inf
        optimal_metrics = {}

        for i in range(num_portfolios):
            weights = self.generate_random_weights(self.num_assets)
            portfolio_return, portfolio_std_dev, sharpe_ratio = self.calculate_portfolio_performance(weights)
            results_matrix[0, i] = portfolio_return
            results_matrix[1, i] = portfolio_std_dev
            results_matrix[2, i] = sharpe_ratio

            if sharpe_ratio > max_sharpe_ratio:
                max_sharpe_ratio = sharpe_ratio
                optimal_weights = weights
                optimal_metrics = {
                    'Max Return': portfolio_return,
                    'Min Return': np.min(self.daily_returns.sum(axis=1)),
                    'Standard Deviation': portfolio_std_dev,
                    'Total Return': portfolio_return,
                    'Annualized Return': (1 + portfolio_return) ** (252 / len(self.daily_returns)) - 1,
                    'Sharpe Ratio': sharpe_ratio,
                    'Optimal Weights': list(optimal_weights)
                }

        return results_matrix, optimal_weights, optimal_metrics

class EfficientFrontierPlotter:
    def __init__(self, optimizer):
        self.results = optimizer.results_matrix
        self.optimal_weights = optimizer.optimal_weights
        self.daily_returns = optimizer.daily_returns

    def plot_efficient_frontier(self):
        optimal_return, optimal_risk, _ = PortfolioOptimizer(self).calculate_portfolio_performance(self.optimal_weights)
        plt.figure(figsize=(14, 8))
        plt.scatter(self.results[1], self.results[0], c=self.results[2], cmap='viridis', marker='o', edgecolor='k', alpha=0.7)
        plt.colorbar(label='Sharpe Ratio')
        plt.title('Efficient Frontier')
        plt.xlabel('Volatility (Standard Deviation)')
        plt.ylabel('Expected Return')
        plt.scatter(optimal_risk, optimal_return, color='red', marker='*', s=200, label='Optimal Risk Adjusted Portfolio')
        plt.legend()
        plt.text(0.02, 0.95, 'The Sharpe Ratio measures the risk-adjusted return of a portfolio.\nThe optimal portfolio maximizes the Sharpe Ratio, providing the best trade-off between return and risk.', 
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        plt.tight_layout()
        plt.show()

class ReportGenerator:
    def __init__(self, etf_data_handler, etf_metrics, optimizer):
        self.etf_data = etf_data_handler.data
        self.etf_metrics = etf_metrics.metrics
        self.optimal_weights = optimizer.optimal_weights
        self.optimal_metrics = optimizer.optimal_metrics

    def generate_report(self, filename):
        with pd.ExcelWriter(filename) as writer:
            self.etf_data.to_excel(writer, sheet_name='Historical ETF Data')
            metrics_data = self.etf_metrics.copy()
            metrics_data['Optimal Weights'] = self.optimal_weights
            metrics_data.to_excel(writer, sheet_name='ETF Metrics')
            optimized_portfolio_metrics = pd.DataFrame([self.optimal_metrics])
            del optimized_portfolio_metrics['Optimal Weights']
            optimized_portfolio_metrics.to_excel(writer, sheet_name='Optimized Portfolio Metrics')

    def condition_metrics_for_percentage(self):
        self.etf_metrics['Total Return (%)'] = self.etf_metrics['Total Return'] * 100
        self.etf_metrics['Annualized Return (%)'] = self.etf_metrics['Annualized Return'] * 100
        self.etf_metrics = self.etf_metrics.drop(columns=['Total Return', 'Annualized Return'])
        self.optimal_metrics['Total Return (%)'] = self.optimal_metrics['Total Return'] * 100
        self.optimal_metrics['Annualized Return (%)'] = self.optimal_metrics['Annualized Return'] * 100
        del self.optimal_metrics['Total Return']
        del self.optimal_metrics['Annualized Return']

# Main function to execute the process
def main():
    etfs = ['QUAL', 'DIA', 'IVV', 'QQQ', 'SOXL']
    # data formats must be: 'YYYY-MM-DD'
    start_date = '2019-01-28'
    end_date = '2024-01-26'
    
    # Step 1: Download Data
    etf_data_handler = ETFDataHandler(etfs, start_date, end_date)
    
    # Step 2: Calculate Metrics
    etf_metrics = ETFMetrics(etf_data_handler)
    
    # Step 3: Optimize Portfolio
    optimizer = PortfolioOptimizer(etf_metrics)
    
    # Step 4: Plot Efficient Frontier
    plotter = EfficientFrontierPlotter(optimizer)
    plotter.plot_efficient_frontier()
    
    # Step 5: Generate Report
    report_generator = ReportGenerator(etf_data_handler, etf_metrics, optimizer)
    report_generator.condition_metrics_for_percentage()
    report_generator.generate_report('etf_report.xlsx')

if __name__ == "__main__":
    main()