import yfinance as yf

def main():
    spx_ticker = '^SPX'
    data = yf.download(spx_ticker, start='1989-09-11', end='2025-05-12')
    data.to_csv('SPX-data.csv')

    spxtr_ticker = '^SP500TR'
    data = yf.download(spxtr_ticker, start='1989-09-11', end='2025-05-12')
    data.to_csv('SPXTR-data.csv')

if __name__ == '__main__':
    main()