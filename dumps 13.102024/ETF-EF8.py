import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
# from scipy.spatial import ConvexHull

class ETFDataHandler:
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.download_data()

    def download_data(self):
        data = yf.download(self.symbols, start=self.start_date, end=self.end_date)['Adj Close']
        print("Data downloaded successfully.")
        # use this to sort the ETF's by name/ticker, and see their start date, end date, and the nr of days of data. 
        # df.groupby("Symbol??")["Date"].agg(['Min', 'Max','count'])
        # then check for null values in the df: df.isnull().sum()
        return data

class ETFMetrics:
    def __init__(self, etf_data_handler):
        self.etf_data = etf_data_handler.data
        self.daily_returns = self.calculate_daily_returns()
        self.metrics = self.calculate_individual_metrics()
        self.correlation_stats = self.calculate_correlation_statistics()

    def calculate_correlation_statistics(self):
        correlation_matrix = self.daily_returns.corr()
        rolling_correlation = self.daily_returns.rolling(window=30).corr().dropna()
        correlation_summary = {
            'Correlation Matrix': correlation_matrix,
            'Mean Correlation': correlation_matrix.mean().mean(),
            'Median Correlation': correlation_matrix.stack().median(),
            'Rolling Correlation (30-day)': rolling_correlation
        }
        return correlation_summary

    def display_correlation_statistics(self):
        print("Correlation Matrix:")
        print(self.correlation_stats['Correlation Matrix'])
        print("\nMean Correlation:", self.correlation_stats['Mean Correlation'])
        print("\nMedian Correlation:", self.correlation_stats['Median Correlation'])
        print("\nRolling Correlation (30-day):")
        print(self.correlation_stats['Rolling Correlation (30-day)'].head())


    def calculate_daily_returns(self):
        daily_returns = self.etf_data.pct_change().dropna() 
        daily_returns = daily_returns.apply(pd.to_numeric, errors='coerce').dropna()
        return daily_returns

    def calculate_individual_metrics(self):
        metrics = pd.DataFrame(index=self.etf_data.columns)
        print(self.daily_returns)
        # selecting the maximum daily return out of the metrics dataframe. A single number per ETF/symbol
        metrics['Max Return'] = self.daily_returns.max()
        # selecting the minimum daily return out of the metrics dataframe. A single number per ETF/symbol
        metrics['Min Return'] = self.daily_returns.min()
        # Annualised standard deviation: std.dev.annual = std.dev.daily * sqrt(nr of trading days). A single number per ETF/symbol
        metrics['Standard Deviation'] = self.daily_returns.std() * np.sqrt(252)  
        # print(metrics['Standard Deviation'])
        # metrics['Real Return'] = (1 + self.daily_returns).cumprod().iloc[-1] - 1 #A single number per ETF/symbol
        metrics['Real Return'] = (self.etf_data.iloc[-1] / self.etf_data.iloc[0]) - 1
        # print(metrics['Real Return'])
        # metrics['Real Return'] = np.exp(np.log(1 + self.daily_returns).sum()) - 1
        metrics['Annualized Return'] = (1 + metrics['Real Return']) ** (252/len(self.daily_returns)) - 1
        # print(f"Lenght of daily returns: {len(self.daily_returns)}")
        # print(metrics['Annualized Return'])
        # metrics['Annualized Return'] = (1 + metrics['Real Return']) ** (1 / (len(self.daily_returns) / 252)) - 1
        metrics['Sharpe Ratio'] = metrics['Annualized Return'] / metrics['Standard Deviation']
        return metrics

class PortfolioOptimizer:
    def __init__(self, etf_metrics, risk_free_rate=0.02):
        self.daily_returns = etf_metrics.daily_returns
        self.num_symbols = len(self.daily_returns.columns)
        self.risk_free_rate = risk_free_rate
        self.max_etf_weight = 1.0  # Maximum weight for a single ETF
        self.weight_step_size = 1  # Weight step size in increments of 1%
        self.results_matrix, self.optimal_weights, self.optimal_metrics, self.all_weights = self.optimize_portfolio(num_portfolios=10000)


    # def generate_random_weights(self, num_symbols):
    #     while True:
    #         weights = np.random.choice(np.arange(0, self.max_etf_weight*100, self.weight_step_size), size=num_symbols)
    #         weights = weights / np.sum(weights)
    #         # check whether all elements in the weights array are less than or equal to self.max_etf_weight.
    #         if np.all(weights <= self.max_etf_weight):
    #             return weights
            
    def generate_random_weights(self, num_symbols):
        while True:
            # Generate random values that sum up to 100
            random_values = np.random.randint(0, 101, size=num_symbols)
            total = np.sum(random_values)

            # Scale the values to sum to 100
            weights = (random_values / total * 100).astype(int)

            # Check if weights sum to 100
            adjustment = 100 - np.sum(weights)
            weights[0] += adjustment  # Adjust the first element to make the sum exactly 100

            # Check the constraint that all weights must be less than or equal to max_etf_weight (expressed in percentage)
            if np.all(weights <= self.max_etf_weight * 100):
                return weights / 100  # Convert to fractions summing to 1.0


    def calculate_portfolio_performance(self, weights):
        # self.daily_returns, df (= "matrix") that contains daily returns, in the shape n,m, where n = nr of trading days, m=number of ETF's
        # weights is a 1,m df (matrix). The DOT product essentially computes a weighted sum of each day's returns for all the ETFs.
        weighted_daily_returns = np.dot(self.daily_returns, weights)
        portfolio_cumulative_return = (1 + weighted_daily_returns).cumprod()[-1] - 1
        portfolio_std_dev = np.std(weighted_daily_returns) * np.sqrt(252)  # Annualize standard deviation
        sharpe_ratio = (portfolio_cumulative_return - self.risk_free_rate) / portfolio_std_dev
        return portfolio_cumulative_return, portfolio_std_dev, sharpe_ratio

    def optimize_portfolio(self, num_portfolios):
        print(f"Simulating {num_portfolios} portfolios. This may take a while.")
        # creating a matrix of 3+self.num_symbols rows, and num_portfolios columns, all filled with 0's. 
        results_matrix = np.zeros((3 + self.num_symbols, num_portfolios))  # Adding an additional row to store weights
        all_weights = []
        optimal_weights = None
        max_sharpe_ratio = -np.inf
        optimal_metrics = {}

        for i in range(num_portfolios):
            weights = self.generate_random_weights(self.num_symbols)
            portfolio_cumulative_return, portfolio_std_dev, sharpe_ratio = self.calculate_portfolio_performance(weights)
            results_matrix[0, i] = portfolio_cumulative_return
            results_matrix[1, i] = portfolio_std_dev
            results_matrix[2, i] = sharpe_ratio
            all_weights.append(weights)
            results_matrix[3:3 + self.num_symbols, i] = weights

            # bubble sorting the max sharp_ratio, optimal weights, optimal metrics. 
            if sharpe_ratio > max_sharpe_ratio:
                max_sharpe_ratio = sharpe_ratio
                optimal_weights = weights
                optimal_metrics = {
                    'Max Return': portfolio_cumulative_return,
                    'Min Return': np.min(self.daily_returns.sum(axis=1)),
                    'Standard Deviation': portfolio_std_dev,
                    'Real Return': portfolio_cumulative_return,
                    'Annualized Return': (1 + portfolio_cumulative_return) ** (len(self.daily_returns)/365) - 1,
                    #(1 + metrics['Real Return']) ** (len(self.daily_returns) / 365) - 1
                    'Sharpe Ratio': sharpe_ratio,
                    'Optimal Weights': list(optimal_weights)
                }

        return results_matrix, optimal_weights, optimal_metrics, all_weights

class EfficientFrontierPlotter:
    def __init__(self, optimizer):
        self.results = optimizer.results_matrix
        self.weights = optimizer.all_weights  # Storing weights for interaction
        self.optimal_weights = optimizer.optimal_weights
        self.optimal_metrics = optimizer.optimal_metrics
        self.daily_returns = optimizer.daily_returns

    def plot_efficient_frontier(self, save_path='efficient_frontier.png'):
        optimal_return = self.optimal_metrics['Max Return']
        optimal_risk = self.optimal_metrics['Standard Deviation']
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('lightgrey')
        ax.grid(True)
        scatter = ax.scatter(self.results[1], self.results[0], c=self.results[2], cmap='RdYlBu', marker='o', edgecolor='k', alpha=0.8, s=30)
        plt.colorbar(scatter, label='Sharpe Ratio')
        ax.set_title('Efficient Frontier')
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Expected Return')
        ax.scatter(optimal_risk, optimal_return, color='green', marker='*', s=300, label='Optimal Risk Adjusted Portfolio')
        
        # Adding marker for least risky portfolio
        min_risk_idx = np.argmin(self.results[1])
        min_risk_return = self.results[0, min_risk_idx]
        min_risk_volatility = self.results[1, min_risk_idx]
        ax.scatter(min_risk_volatility, min_risk_return, color='blue', marker='*', s=200, label='Least Risky Portfolio')
        ax.legend()
        ax.text(0.02, 0.90, 'The Sharpe Ratio measures the risk-adjusted return of a portfolio.\n'
                 'The optimal portfolio maximizes the Sharpe Ratio,\n providing the best trade-off between return and risk.', 
                 transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        
        # Adding interactive cursor to show weights
        cursor = mplcursors.cursor(scatter, hover=True)
        
        @cursor.connect("add")
        def on_hover(sel):
            index = sel.index
            sel.annotation.set_text(f"Weights: {self.weights[index]}\nReturn: {self.results[0, index]:.2f}\nStd Dev: {self.results[1, index]:.2f}")
            sel.annotation.get_bbox_patch().set(facecolor='black', alpha=0.75)  # Set hover box color to black
            sel.annotation.set_color('white')  # Set text color to white
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()


# class EfficientFrontierPlotter:
#     def __init__(self, optimizer):
#         self.results = optimizer.results_matrix
#         self.weights = optimizer.all_weights  # Storing weights for interaction
#         self.optimal_weights = optimizer.optimal_weights
#         self.daily_returns = optimizer.daily_returns

#     def plot_efficient_frontier(self, save_path='efficient_frontier.png'):
#         optimal_return, optimal_risk, _ = PortfolioOptimizer(self).calculate_portfolio_performance(self.optimal_weights)
#         fig, ax = plt.subplots(figsize=(16, 10))
#         fig.patch.set_facecolor('white')
#         ax.set_facecolor('lightgrey')
#         ax.grid(True)
#         scatter = ax.scatter(self.results[1], self.results[0], c=self.results[2], cmap='RdYlBu', marker='o', edgecolor='k', alpha=0.8, s=30)
#         plt.colorbar(scatter, label='Sharpe Ratio')
#         ax.set_title('Efficient Frontier')
#         ax.set_xlabel('Volatility (Standard Deviation)')
#         ax.set_ylabel('Expected Return')
#         ax.scatter(optimal_risk, optimal_return, color='red', marker='*', s=300, label='Optimal Risk Adjusted Portfolio')
        
#         # Adding marker for least risky portfolio
#         min_risk_idx = np.argmin(self.results[1])
#         min_risk_return = self.results[0, min_risk_idx]
#         min_risk_volatility = self.results[1, min_risk_idx]
#         ax.scatter(min_risk_volatility, min_risk_return, color='blue', marker='o', s=200, label='Least Risky Portfolio')
#         ax.legend()
#         ax.text(0.02, 0.90, 'The Sharpe Ratio measures the risk-adjusted return of a portfolio.\n'
#                  'The optimal portfolio maximizes the Sharpe Ratio,\n providing the best trade-off between return and risk.', 
#                  transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        
#         # Adding interactive cursor to show weights
#         cursor = mplcursors.cursor(scatter, hover=True)
        
#         @cursor.connect("add")
#         def on_hover(sel):
#             index = sel.index
#             sel.annotation.set_text(f"Weights: {self.weights[index]}")
#             sel.annotation.get_bbox_patch().set(facecolor='black', alpha=0.75)  # Set hover box color to black
#             sel.annotation.set_color('white')  # Set text color to white
        
#         plt.tight_layout()
#         plt.savefig(save_path)
#         plt.show()

class ReportGenerator:
    def __init__(self, etf_data_handler, etf_metrics, optimizer, results, plot_path='efficient_frontier.png'):
        self.etf_data = etf_data_handler.data
        self.etf_metrics = etf_metrics
        self.optimal_weights = optimizer.optimal_weights
        self.optimal_metrics = optimizer.optimal_metrics
        self.results = results
        self.plot_path = plot_path
        self.constraints = {
            'Max ETF Weight': optimizer.max_etf_weight,
            'Weight Step Size': optimizer.weight_step_size
        }

    def generate_report(self, filename,):
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            print(self.etf_data)
            self.etf_data.to_excel(writer, sheet_name='Historical ETF Data')
            metrics_data = self.etf_metrics.metrics.copy()
            metrics_data['Optimal Weights'] = self.optimal_weights
            metrics_data.to_excel(writer, sheet_name='ETF Metrics')
            
            # Add Correlation Statistics to the report
            correlation_matrix = self.etf_metrics.correlation_stats['Correlation Matrix']
            correlation_matrix.to_excel(writer, sheet_name='Correlation Matrix')
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
        self.etf_metrics.metrics['Real Return (%)'] = self.etf_metrics.metrics['Real Return'] * 100
        self.etf_metrics.metrics['CAGR (%)'] = self.etf_metrics.metrics['Annualized Return'] * 100
        self.etf_metrics.metrics = self.etf_metrics.metrics.drop(columns=['Real Return', 'Annualized Return'])
        self.optimal_metrics['Real Return (%)'] = self.optimal_metrics['Real Return'] * 100
        self.optimal_metrics['CAGR (%)'] = self.optimal_metrics['Annualized Return'] * 100
        del self.optimal_metrics['Real Return']
        del self.optimal_metrics['Annualized Return']

# Main function to execute the process
def main():
    asset_list = [
        ['HD','MSFT','TMO','FMG.AX'],
        ['IYW', 'COPX', 'SPUU', 'XLK', 'SSO', 'ITB', 'AIRR', 'FTEC', 'XHB', 'VGT', 'FCG', 'IXN', 'XSD', 'PAVE', 'GRN', 'PTF', 'IGM', 'GRID', 'QQQ', 'XME'],
        ['IYW', 'COPX', 'XLK', 'VGT','FTEC'],
        ['IYW', 'COPX', 'SPUU', 'SSO','GRN'],
        ['IVV', 'TDT.AS', 'EWD', 'QUAL','SOXL'],
        ['SSO', 'PAVE', 'XHB', 'GRID', 'XME', 'IYW', 'COPX'],
        ['AAPL', 'AMZN', 'MSFT'],
        ['AAPL', 'AMZN', 'MSFT', 'GOOGL', 'TSLA', 'NVDA','DIS', 'NFLX', 'CMCSA', 'T'],
    ]

    # Select one of the sets to experiment with
    symbols = asset_list[15]
    start_date = '2014-01-01'
    end_date = '2024-01-01'
    # end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    # Step 1: Download Data
    etf_data_handler = ETFDataHandler(symbols, start_date, end_date)

    # Step 2: Calculate Metrics
    etf_metrics = ETFMetrics(etf_data_handler)
    
    # Display Correlation Statistics
    etf_metrics.display_correlation_statistics()

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




# example ETF's

# iShares U.S. Technology ETF (IYW) - 25.45%
# Global X Copper Miners ETF (COPX) - 25.30%
# Direxion Daily S&P 500 Bull 2x Shares (SPUU) - 25.03%
# Technology Select Sector SPDR Fund (XLK) - 24.63%
# ProShares Ultra S&P 500 (SSO) - 24.49%
# iShares U.S. Home Construction ETF (ITB) - 24.18%
# First Trust RBA American Industrial Renaissance ETF (AIRR) - 24.04%
# Fidelity MSCI Information Technology Index ETF (FTEC) - 23.97%
# SPDR S&P Homebuilders ETF (XHB) - 23.90%
# Vanguard Information Technology ETF (VGT) - 23.79%
# First Trust Natural Gas ETF (FCG) - 23.63%
# iShares Global Tech ETF (IXN) - 22.85%
# SPDR S&P Semiconductor ETF (XSD) - 22.86%
# Global X US Infrastructure Development ETF (PAVE) - 22.57%
# iPath Series B Carbon Exchange-Traded Notes (GRN) - 22.53%
# Invesco Dorsey Wright Technology Momentum ETF (PTF) - 23.40%
# iShares Expanded Tech Sector ETF (IGM) - 22.64%
# First Trust Nasdaq Clean Edge Smart GRID Infrastructure Index (GRID) - 22.37%
# Invesco QQQ Trust Series I (QQQ) - 21.99%
# SPDR S&P Metals & Mining ETF (XME) - 21.93%