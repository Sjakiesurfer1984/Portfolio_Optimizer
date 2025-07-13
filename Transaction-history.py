from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import csv
import time

class IBKRApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.transaction_data = []
        self.reqId = 0  # Request ID for transaction requests

    def historicalData(self, reqId, date, open, high, low, close, volume, closeAdj, openAdj, highAdj, lowAdj, volumeAdj, barCount, hasGaps):
        transaction_details = {
            'Date': date,
            'Open': open,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume,
        }
        self.transaction_data.append(transaction_details)

    def historicalDataEnd(self, reqId):
        print("Historical data retrieval complete.")
        if self.transaction_data:
            print(f"Retrieved {len(self.transaction_data)} transaction records.")
            self.save_to_csv()
        else:
            print("No transaction data retrieved.")

    def save_to_csv(self):
        output_file = "C:\\Users\\timvo\\OneDrive\\Documents\\Belasting\\ato_transaction_report.csv"
        
        if self.transaction_data:
            with open(output_file, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.transaction_data[0].keys())
                writer.writeheader()
                writer.writerows(self.transaction_data)
            print(f"Transaction data saved to {output_file}")
        else:
            print("No data to save.")

def download_transaction_history(app, start_date, end_date):
    contract = Contract()
    contract.symbol = "AAPL"  # Example: Apple Inc.
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"

    # Log contract details
    print(f"Requesting historical data for {contract.symbol}")

    # Adjusting date format
    app.reqHistoricalData(app.reqId, contract, end_date, "1 M", "1 day", "TRADES", 1, 1, False, [])
    app.run()  # This will call historicalData when data is received

def download_data_for_ato():
    app = IBKRApp()
    app.connect('127.0.0.1', 7496, 0)  # Connect to TWS on port 7496

    # Wait for connection to complete
    time.sleep(1)

    # Define the date range for the fiscal year in UTC format
    start_date = "20230701 00:00:00"  # July 1, 2023
    end_date = "20240630 23:59:59"     # June 30, 2024

    download_transaction_history(app, start_date, end_date)  # Start the download process

if __name__ == "__main__":
    download_data_for_ato()  # Run the data download function
