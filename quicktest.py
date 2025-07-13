import yfinance as yf

ticker = yf.Ticker("AAPL")
hist = yf.Ticker("AAPL").history(period="1mo")
print(hist)

print(hist.head())
