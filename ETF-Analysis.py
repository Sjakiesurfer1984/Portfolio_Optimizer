import requests
import pandas as pd

class ETFDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3/quote/"
        self.etfs_list = []

    def get_top_20_etfs(self):
        # This would ideally pull data from the API to get the top 20 performing ETFs.
        # For now, we use a placeholder list of symbols.
        self.etfs_list = [
            'IYW', 'COPX', 'SPUU', 'XLK', 'SSO', 'ITB', 'AIRR', 'FTEC', 'XHB', 'VGT',
            'FCG', 'IXN', 'XSD', 'PAVE', 'GRN', 'PTF', 'IGM', 'GRID', 'QQQ', 'XME'
        ]

    def fetch_etf_data(self):
        etf_details = []
        for etf in self.etfs_list:
            url = f"{self.base_url}{etf}?apikey={self.api_key}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()[0] if isinstance(response.json(), list) else response.json()
                etf_details.append({
                    'MANAGING COMPANY': data.get('companyName', 'N/A'),
                    'ETF SYMBOL': etf,
                    'STRATEGY': data.get('investment_strategy', 'N/A'),
                    'INCEPTION DATE': data.get('inception_date', 'N/A'),
                    'Domicile': data.get('country', 'N/A'),
                    'DOMINANT COUNTRY OF OPERATION': data.get('country', 'N/A'),
                    'PRICE': data.get('price', 'N/A'),
                    'Management fee [p.a]': data.get('expense_ratio', 'N/A'),
                    'PERFORMANCE FEE': data.get('performance_fee', 'N/A'),
                    'FUND SIZE': data.get('marketCap', 'N/A'),
                    'DIVIDEND FREQUENCY': data.get('dividend_frequency', 'N/A'),
                    'DIV. YIELD': data.get('dividendYield', 'N/A'),
                    'ANNUAL RETURN': data.get('annual_return', 'N/A'),
                    '# HOLDINGS': data.get('holding_count', 'N/A'),
                    'INDUSTRIES THE ETF OPERATES IN, INCLUDING WEIGHTS [%]': data.get('sector_weightings', 'N/A')
                })
            else:
                print(f"Failed to fetch data for ETF: {etf}")
        return etf_details

    def save_to_excel(self, etf_details):
        df = pd.DataFrame(etf_details)
        df.to_excel('top_20_etfs_report.xlsx', index=False)

if __name__ == "__main__":
    api_key = "YOUR_FINANCIAL_MODELING_PREP_API_KEY"
    collector = ETFDataCollector(api_key)
    collector.get_top_20_etfs()
    etf_details = collector.fetch_etf_data()
    collector.save_to_excel(etf_details)
    print("Report generated: top_20_etfs_report.xlsx")
