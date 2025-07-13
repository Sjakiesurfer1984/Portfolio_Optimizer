import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from scipy.spatial import ConvexHull

class ETFDataHandler:
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.download_data()

    def download_data(self):
        data = yf.download(self.symbols, start=self.start_date, end=self.end_date)['Adj Close']
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
    def __init__(self, etf_metrics, risk_free_rate=0.02):
        self.daily_returns = etf_metrics.daily_returns
        self.num_symbols = len(self.daily_returns.columns)
        self.risk_free_rate = risk_free_rate
        self.max_etf_weight = 0.4  # Maximum weight for a single ETF
        self.weight_step_size = 0.01  # Weight step size in increments of 1%
        self.results_matrix, self.optimal_weights, self.optimal_metrics, self.all_weights = self.optimize_portfolio(num_portfolios=10000)

    def generate_random_weights(self, num_symbols):
        while True:
            weights = np.random.choice(np.arange(0, 1 + self.weight_step_size, self.weight_step_size), size=num_symbols)
            weights = weights / np.sum(weights)
            if np.all(weights <= self.max_etf_weight):
                return weights

    def calculate_portfolio_performance(self, weights):
        weighted_returns = np.dot(self.daily_returns, weights)
        portfolio_return = (1 + weighted_returns).cumprod()[-1] - 1
        portfolio_std_dev = np.std(weighted_returns) * np.sqrt(252)  # Annualize standard deviation
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std_dev
        return portfolio_return, portfolio_std_dev, sharpe_ratio

    def optimize_portfolio(self, num_portfolios):
        print(f"Simulating {num_portfolios} portfolios. This may take a while.")
        results_matrix = np.zeros((3 + self.num_symbols, num_portfolios))  # Adding an additional row to store weights
        all_weights = []
        optimal_weights = None
        max_sharpe_ratio = -np.inf
        optimal_metrics = {}

        for i in range(num_portfolios):
            weights = self.generate_random_weights(self.num_symbols)
            portfolio_return, portfolio_std_dev, sharpe_ratio = self.calculate_portfolio_performance(weights)
            results_matrix[0, i] = portfolio_return
            results_matrix[1, i] = portfolio_std_dev
            results_matrix[2, i] = sharpe_ratio
            all_weights.append(weights)
            results_matrix[3:3 + self.num_symbols, i] = weights

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

        return results_matrix, optimal_weights, optimal_metrics, all_weights

class EfficientFrontierPlotter:
    def __init__(self, optimizer):
        self.results = optimizer.results_matrix
        self.weights = optimizer.all_weights  # Storing weights for interaction
        self.optimal_weights = optimizer.optimal_weights
        self.daily_returns = optimizer.daily_returns

    def plot_efficient_frontier(self, save_path='efficient_frontier.png'):
        optimal_return, optimal_risk, _ = PortfolioOptimizer(self).calculate_portfolio_performance(self.optimal_weights)
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('lightgrey')
        ax.grid(True)
        scatter = ax.scatter(self.results[1], self.results[0], c=self.results[2], cmap='RdYlBu', marker='o', edgecolor='k', alpha=0.8, s=30)
        plt.colorbar(scatter, label='Sharpe Ratio')
        ax.set_title('Efficient Frontier')
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.scatter(optimal_risk, optimal_return, color='red', marker='*', s=300, label='Optimal Risk Adjusted Portfolio')
        
        # Adding marker for least risky portfolio
        min_risk_idx = np.argmin(self.results[1])
        min_risk_return = self.results[0, min_risk_idx]
        min_risk_volatility = self.results[1, min_risk_idx]
        ax.scatter(min_risk_volatility, min_risk_return, color='blue', marker='o', s=200, label='Least Risky Portfolio')
        ax.legend()
        ax.text(0.02, 0.90, 'The Sharpe Ratio measures the risk-adjusted return of a portfolio.\n'
                 'The optimal portfolio maximizes the Sharpe Ratio,\n providing the best trade-off between return and risk.', 
                 transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        
        # Adding interactive cursor to show weights
        cursor = mplcursors.cursor(scatter, hover=True)
        
        @cursor.connect("add")
        def on_hover(sel):
            index = sel.index
            sel.annotation.set_text(f"Weights: {self.weights[index]}")
            sel.annotation.get_bbox_patch().set(facecolor='black', alpha=0.75)  # Set hover box color to black
            sel.annotation.set_color('white')  # Set text color to white
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

class ReportGenerator:
    def __init__(self, etf_data_handler, etf_metrics, optimizer, results, plot_path='efficient_frontier.png'):
        self.etf_data = etf_data_handler.data
        self.etf_metrics = etf_metrics.metrics
        self.optimal_weights = optimizer.optimal_weights
        self.optimal_metrics = optimizer.optimal_metrics
        self.results = results
        self.plot_path = plot_path
        self.constraints = {
            'Max ETF Weight': optimizer.max_etf_weight,
            'Weight Step Size': optimizer.weight_step_size
        }

    def generate_report(self, filename):
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            print(self.etf_data)
            self.etf_data.to_excel(writer, sheet_name='Historical ETF Data')
            metrics_data = self.etf_metrics.copy()
            metrics_data['Optimal Weights'] = self.optimal_weights
            metrics_data.to_excel(writer, sheet_name='ETF Metrics')
            optimized_portfolio_metrics = pd.DataFrame([self.optimal_metrics])
            del optimized_portfolio_metrics['Optimal Weights']
            optimized_portfolio_metrics.to_excel(writer, sheet_name='Optimized Portfolio Metrics', startrow=0, startcol=0)

            # Add constraints to the worksheet
            worksheet = writer.sheets['Optimized Portfolio Metrics']
            worksheet.write('A15', 'Constraints:')
            worksheet.write('A16', 'Max ETF Weight')
            worksheet.write('B16', self.constraints['Max ETF Weight'])
            worksheet.write('A17', 'Weight Step Size')
            worksheet.write('B17', self.constraints['Weight Step Size'])

            # Access the XlsxWriter workbook and worksheet objects
            workbook = writer.book

            # Insert the saved plot image into the worksheet
            worksheet.insert_image('G2', self.plot_path)
            print(f"Saving {filename}")

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
    asset_list = [
        ['IYW', 'COPX', 'XLK', 'VGT','FTEC'],
        ['IYW', 'COPX', 'SPUU', 'SSO','GRN'],
        ['IVV', 'TDT.AS', 'EWD', 'QUAL','SOXL'],
        ['AAPL', 'AMZN', 'MSFT'],
        ['AAPL', 'AMZN', 'MSFT', 'GOOGL', 'TSLA', 'NVDA','DIS', 'NFLX', 'CMCSA', 'T'],
        ['GOOGL', 'TSLA', 'NVDA'],
        ['JPM', 'BAC', 'C', 'WFC', 'GS'],
        ['DIS', 'NFLX', 'CMCSA', 'T'],
        ['XOM', 'CVX', 'BP', 'TOT'],
        ['NKE', 'SBUX', 'MCD', 'TGT', 'WMT'],
        ['F', 'GM', 'RIVN', 'NIO'],
        ['PG', 'KO', 'PEP', 'JNJ', 'MRK'],
        ['CAT', 'GE', 'MMM', 'HON'],
        ['WBA', 'CVS', 'TGT', 'COST'],
        ['VTI', 'VEA', 'BND', 'VNQ', 'VOO'],
        ['IWM', 'EFA', 'TLT', 'TIP', 'DBC']
    ]

    # Select one of the sets to experiment with
    symbols = asset_list[1]
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    # end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    # Step 1: Download Data
    etf_data_handler = ETFDataHandler(symbols, start_date, end_date)

    # Step 2: Calculate Metrics
    etf_metrics = ETFMetrics(etf_data_handler)

    # Step 3: Optimize Portfolio
    risk_free_rate = 0.02  # Set the risk-free rate (e.g., 2%)
    optimizer = PortfolioOptimizer(etf_metrics, risk_free_rate)

    # Step 4: Plot Efficient Frontier
    plotter = EfficientFrontierPlotter(optimizer)
    plotter.plot_efficient_frontier(save_path='efficient_frontier.png')

    # Step 5: Generate Report
    report_generator = ReportGenerator(etf_data_handler, etf_metrics, optimizer, optimizer.results_matrix, 'efficient_frontier.png')
    report_generator.condition_metrics_for_percentage()
    report_generator.generate_report('etf_report.xlsx')
    print("Exiting program")

if __name__ == "__main__":
    main()