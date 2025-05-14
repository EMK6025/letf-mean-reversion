import yfinance as yf

ticker = "^SPX"

# Download historical data (adjust the start/end dates as needed)
data = yf.download(ticker, start="1989-09-11", end="2025-05-12")

data.to_csv("SPX-data.csv")
