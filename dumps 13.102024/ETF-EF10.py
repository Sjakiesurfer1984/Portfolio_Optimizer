import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from pprint import pprint
import time
from datetime import datetime

class ETFDataHandler:
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.download_data()

    def download_data(self):
        print("starting data download...")
        data = yf.download(self.symbols, start=self.start_date, end=self.end_date)['Adj Close']
        print("Data downloaded successfully.")
        self.verify_shortest_time_range(data)
        return self.data

    def verify_shortest_time_range(self, data):
        print("Verifying shortest time range available for each ticker symbol...")
        min_dates = data.apply(lambda x: x.first_valid_index())
        min_date = min_dates.max()
        if min_date and pd.Timestamp(min_date) > pd.Timestamp(self.start_date):
            print(f"Adjusting start date from {self.start_date} to {min_date} to ensure equal time range for all ETFs.")
            self.start_date = min_date
            # Truncate the data to ensure equal time range for all ETFs
            self.data = data[data.index >= min_date]
        else:
            self.data = data


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
        # print(self.daily_returns)
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
    def __init__(self, etf_metrics, risk_free_rate=0.03):
        # Ensure max_etf_weight is feasible for the number of ETFs
        self.num_symbols = len(etf_metrics.daily_returns.columns)
        self.max_etf_weight = 0.3  # Maximum weight for a single ETF
        self.weight_step_size = 0.1  # Weight step size in increments of 1%
        min_required_max_weight = 1.0 / self.num_symbols * 1.2
        if self.max_etf_weight < min_required_max_weight:
            print(f"Adjusting max_etf_weight from {self.max_etf_weight} to {min_required_max_weight} to ensure feasibility.")
            self.max_etf_weight = min_required_max_weight
        self.daily_returns = etf_metrics.daily_returns
        self.num_symbols = len(self.daily_returns.columns)
        self.risk_free_rate = risk_free_rate
        self.results_matrix, self.optimal_weights, self.optimal_metrics, self.all_weights = self.optimize_portfolio(num_portfolios=10000)
           
    def generate_random_weights(self, num_symbols):
        start_time = time.time()
        while True:
            # Exit the loop if 10 seconds have passed
            if time.time() - start_time > 10:
                print("Exiting weight generation due to timeout.")
                return None

            # Generate weights using Dirichlet distribution
            weights = np.random.dirichlet(np.ones(num_symbols), size=1)[0]

            # Check that none of the weights exceed the max allowed weight constraint
            if np.all(weights <= self.max_etf_weight):
                return weights

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
                    'Annualized Return': (1 + portfolio_cumulative_return) ** (1/(len(self.daily_returns)/252)) - 1, # 252 days, because on average there are 252 trading days in a year in the US. 
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
        optimal_return = self.optimal_metrics['Max Return']+1
        optimal_risk = self.optimal_metrics['Standard Deviation']
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('lightgrey')
        ax.grid(True)
        scatter = ax.scatter(self.results[1], self.results[0]+1, c=self.results[2], cmap='RdYlBu', marker='o', edgecolor='k', alpha=0.8, s=30)
        plt.colorbar(scatter, label='Sharpe Ratio')
        ax.set_title('Efficient Frontier')
        ax.set_xlabel('Volatility (Standard Deviation)')
        ax.set_ylabel('Portfolio Return [x times]')
        ax.scatter(optimal_risk, optimal_return, color='green', marker='*', s=400, label='Optimal Risk Adjusted Portfolio')
        
        # Adding marker for least risky portfolio
        min_risk_idx = np.argmin(self.results[1])
        min_risk_return = self.results[0, min_risk_idx]+1
        min_risk_volatility = self.results[1, min_risk_idx]
        ax.scatter(min_risk_volatility, min_risk_return, color='blue', marker='*', s=300, label='Least Risky Portfolio')
        ax.legend()
        # ax.text(0.02, 0.90, 'The Sharpe Ratio measures the risk-adjusted return of a portfolio.\n'
        #          'The optimal portfolio maximizes the Sharpe Ratio,\n providing the best trade-off between return and risk.', 
        #          transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        
        ax.text(0, -0.2, 'The Sharpe Ratio measures the risk-adjusted return of a portfolio.\n'
                 'The optimal portfolio maximizes the Sharpe Ratio,\nproviding the best trade-off between return and risk.', transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))

        # Adding interactive cursor to show weights
        cursor = mplcursors.cursor(scatter, hover=True)
        
        @cursor.connect("add")
        def on_hover(sel):
            index = sel.index
            # sel.annotation.set_text(f"Weights: {self.weights[index]}\nReturn: {(self.results[0, index]+1):.2f}\nStd Dev: {self.results[1, index]:.2f}")
            sel.annotation.set_text(f"Weights: {self.weights[index]}\nReturn: {(self.results[0, index] + 1):.2f}X\nStd Dev: {self.results[1, index]:.2f}")
            sel.annotation.get_bbox_patch().set(facecolor='black', alpha=0.75)  # Set hover box color to black
            sel.annotation.set_color('white')  # Set text color to white
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()


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

    def prepare_report_data(self):
        report_data = {
            'Historical ETF Data': self.etf_data,
            'ETF Metrics': self.etf_metrics.metrics.copy(),
            'Correlation Matrix': self.etf_metrics.correlation_stats['Correlation Matrix'],
            'Mean Correlation': self.etf_metrics.correlation_stats['Mean Correlation'],
            'Median Correlation': self.etf_metrics.correlation_stats['Median Correlation'],
            'Optimized Portfolio Metrics': pd.DataFrame([self.optimal_metrics]),
            'Constraints': self.constraints
        }
        report_data['ETF Metrics']['Optimal Weights'] = self.optimal_weights
        del report_data['Optimized Portfolio Metrics']['Optimal Weights']
        return report_data
    
    def get_report_filename(self):
        portfolio_str = '_'.join(self.etf_data.columns)
        current_date = datetime.now().strftime('%Y-%m-%d')
        return f'etf_report_{portfolio_str}_{current_date}.xlsx'

    def save_report(self, report_data, filename):
        if filename is None:
            filename = self.get_report_filename()

        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            print(self.etf_data)
            report_data['Historical ETF Data'].to_excel(writer, sheet_name='Historical ETF Data')
            report_data['ETF Metrics'].to_excel(writer, sheet_name='ETF Metrics')
            report_data['Correlation Matrix'].to_excel(writer, sheet_name='Correlation Matrix')

            report_data['Optimized Portfolio Metrics'].to_excel(writer, sheet_name='Optimized Portfolio Metrics', startrow=0, startcol=0)

            # Add constraints to the worksheet
            worksheet = writer.sheets['Optimized Portfolio Metrics']
            worksheet.write('A5', 'Constraints:')
            worksheet.write('A6', 'Max ETF Weight')
            worksheet.write('B6', report_data['Constraints']['Max ETF Weight'])
            worksheet.write('A7', 'Weight Step Size')
            worksheet.write('B7', report_data['Constraints']['Weight Step Size'])

            # Insert the saved plot image into the worksheet
            worksheet.insert_image('G2', self.plot_path)

            # Add more statistics to the worksheet
            worksheet = writer.sheets['Correlation Matrix']
            worksheet.write('A9', 'Mean Correlation')
            worksheet.write('B9', report_data['Mean Correlation'])
            worksheet.write('A10', 'Median Correlation')
            worksheet.write('B10', report_data['Median Correlation'])

            # Access the XlsxWriter workbook and worksheet objects
            workbook = writer.book

            print(f"Saving {filename}")

    def condition_metrics_for_percentage(self):
        #### Individual ETF Metrics ####
        self.etf_metrics.metrics['Real Return (%)'] = self.etf_metrics.metrics['Real Return'] * 100
        self.etf_metrics.metrics['CAGR (%)'] = self.etf_metrics.metrics['Annualized Return'] * 100
        self.etf_metrics.metrics = self.etf_metrics.metrics.drop(columns=['Real Return', 'Annualized Return'])
        #### Portfolio Metrics ####
        self.optimal_metrics['Real Return (%)'] = self.optimal_metrics['Real Return'] * 100 
        self.optimal_metrics['CAGR (%)'] = self.optimal_metrics['Annualized Return'] * 100 
        del self.optimal_metrics['Real Return']
        del self.optimal_metrics['Annualized Return']

# Main function to execute the process
def main():
    asset_list = [
        ['SMH','IHI','XBI','ITB','SOXL'], # Line 0
        ['SPY','QQQ','VTI','VWO','VEA','QUAL'], # Line 1, etc. 
        ["TAN", "SMH", "ARGT", "IYW", 'IHI'], #2
        ["SOXL", "SMH", "ARGT", "IYW", 'IHI'], #3
        ["QUAL", "SMH", "ARGT", "IYW", 'IHI'], #4
        ["QUAL", "SMH", "ARGT", "IYW", 'IHI', 'SOXL'], #5
        ["EEM", "SMH", "ARGT", "IYW", 'IHI', 'SOXL'], #6
        ["TAN", "SMH", "ARGT", "IYW", "SPY", "EFA", "EEM", "AGG", "GLD", "VNQ"], #7
        ['SMH','IHI','XBI','QUAL','SOXL'], #8
        ['SMH','IHI','XBI','ITB','SOXL'], #9
        ['SMH','IHI','HODL','SOXL', 'QUAL'], #10
        ['SMH','IHI','ITB','SOXL', 'QUAL'],#11
        ['IYW', 'COPX', 'SPUU', 'XLK', 'SSO', 'ITB', 'AIRR', 'FTEC', 'XHB', 'VGT', 'FCG', 'IXN', 'XSD', 'PAVE', 'GRN', 'PTF', 'IGM', 'GRID', 'QQQ', 'XME'], #12
        ['IYW', 'COPX', 'XLK', 'VGT','FTEC'], #13
        ['IYW', 'COPX', 'SPUU', 'SSO','GRN'], #14
        ['IVV', 'TDT.AS', 'EWD', 'QUAL','SOXL'], #15
        ['SSO', 'PAVE', 'XHB', 'GRID', 'XME', 'IYW', 'COPX'], #16
        ['AAPL', 'AMZN', 'MSFT'], #17
        ['AAPL', 'AMZN'],
        ['AAPL'],
        ['AAPL', 'AMZN', 'MSFT', 'GOOGL', 'TSLA', 'NVDA','DIS', 'NFLX', 'CMCSA', 'T'],
    ]

    # Select one of the sets to experiment with
    symbols = asset_list[6]
    start_date = '2014-11-30'
    end_date = '2024-11-30'
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
    report_data = report_generator.prepare_report_data()
    # report_generator.save_report(report_data, 'etf_report.xlsx')
    report_generator.save_report(report_data, filename = None)
    print("Exiting program")

if __name__ == "__main__":
    main()


# Step 1: Include the symbols you'd like to analyse in the asset_list = [] list. Every line (= ['ETF1', 'ETF2'],) is a portfolio.
# Step 2:  symbols = asset_list[0], 0 indicates it will analyse row 0 of the list that you populated in step 1. Likewise, 1 means the programme will analyse the portfolio you defined on line 1, etc. 
# Step 3: OPTIONAL: self.max_etf_weight = 0.3  # Maximum weight for a single ETF. 0.3 corresponds to 30%, meaning a single ETF of the portfolio you defined in step 1, cannot exceed a maximum weight of 30%.
# and self.weight_step_size = 1  # Weight step size in increments of 1%. The programme calculated returns and risk for every possible combination of portfolio compositions in steps of 1%. ALterantively, you can set it to larger steps. 
# Step 4: Set the risk-free rate 

# To do:
# request user where to save the report.
# Suggest more descriptive file name: Porfolio analysis ETF1 ETF2 etc.xslx
# add check to see if ETF ticker exists. If not, prompt user to enter ticker correctly OR skip ticker. 
# Add check to determine if there is data for ALL ETF's for the stated period. If NOT:
    # determine the longest period available. Find MAX of all ETF's and compare. 

# Add top 10 of max sharpe ratio portfolios to analysis sheet + lowest risk + highest return portfolio, inculding their metrics. 
# Create GUI, such that the user can reload the portfolio and hover over the graph to see all sorts of portfolio compositions, their returns and risks. 
# 
# Create executable file.
