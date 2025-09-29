import yfinance as yf

# Choose your ticker, e.g., SPXTR (S&P 500 Total Return Index)
ticker = '^SP500TR'  # Use '^' for indices on Yahoo

# Download historical data (adjust the start/end dates as needed)
data = yf.download(ticker, start='1989-09-11', end='2025-05-12')

data.to_csv('SPXTR-data.csv')
