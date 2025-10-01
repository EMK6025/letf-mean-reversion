import pandas as pd
from pathlib import Path

def main():
    # define paths
    proj_path = Path(__file__).resolve().parent.parent
    spx_path = proj_path / 'SPX-data.csv'
    spxtr_path = proj_path / 'SPXTR-data.csv'
    rf_path = proj_path / 'F-F_Research_Data_Factors_daily.csv'
    sso_path = proj_path / 'SSO-historical_nav.csv'
    upro_path = proj_path / 'UPRO-historical_nav.csv'
    
    
    # load data
    spx = pd.read_csv(spx_path, usecols=['Date', 'SPX Close'])
    spxtr = pd.read_csv(spxtr_path, usecols=['Date', 'SPXTR Close'])
    rf = pd.read_csv(rf_path, usecols=['Date', 'RF Rate'])
    
    
    # process spx, spxtr, and rf
    spx['Date'] = pd.to_datetime(spx['Date'], dayfirst=False).dt.date
    spx['SPX Close'] = spx['SPX Close'].astype(float)
    spx = spx.set_index('Date').sort_index()
    
    spxtr['Date'] = pd.to_datetime(spxtr['Date'], dayfirst=False).dt.date
    spxtr['SPXTR Close'] = spxtr['SPXTR Close'].astype(float)
    spxtr = spxtr.set_index('Date').sort_index()

    rf['Date'] = pd.to_datetime(rf['Date'], dayfirst=False).dt.date
    rf['RF Rate'] = rf['RF Rate'].astype(float)
    rf = rf.set_index('Date').sort_index()
    
    
    # store data
    combined = spx.join([spxtr, rf], how='inner')
    combined = combined.reset_index()
    combined.to_csv(proj_path / 'combined_data.csv', index=False)


    # grab and process sso
    sso = pd.read_csv(sso_path, usecols=['Date', 'NAV Change (%)'])
    sso = sso.rename(columns={
        'NAV Change (%)': '2x LETF Change'
    })
    sso['Date'] = pd.to_datetime(sso['Date'])
    sso['2x LETF Change'] = sso['2x LETF Change'].astype(float)
    sso = sso.set_index('Date')


    # grab and process upro
    upro = pd.read_csv(upro_path, usecols=['Date', 'NAV Change (%)'])
    upro = upro.rename(columns={
        'NAV Change (%)': '3x LETF Change'
    })
    upro['Date'] = pd.to_datetime(upro['Date'])
    upro['3x LETF Change'] = upro['3x LETF Change'].astype(float)
    upro = upro.set_index('Date').sort_index()
    
    
    # store data
    combined = sso.join(upro, how='inner')
    combined = combined.reset_index()
    
    combined.to_csv(proj_path / 'LETF_data.csv', index=False)

    

if __name__ == '__main__':
    main()
    
# spx_path = Path(__file__).resolve().parent.parent / 'SPX-data.csv'
# df = pd.read_csv(spx_path, usecols=['Date', 'Close'])
# df['Date'] = pd.to_datetime(df['Date'], dayfirst=False).dt.date
# df['Close'] = df['Close'].astype(float)
# df = df.sort_values('Date')
# df = df.rename(columns={
#     'Close': 'SPX Close'
# })
# df.to_csv(spx_path, index=False)

# spxtr_path = Path(__file__).resolve().parent.parent / 'SPXTR-data.csv'
# df = pd.read_csv(spxtr_path, usecols=['Date', 'Close'])
# df['Date'] = pd.to_datetime(df['Date'], dayfirst=False).dt.date
# df['Close'] = df['Close'].astype(float)
# df = df.sort_values('Date')
# df = df.rename(columns={
#     'Close': 'SPXTR Close'
# })
# df.to_csv(spxtr_path, index=False)

# rf_path = Path(__file__).resolve().parent.parent / 'F-F_Research_Data_Factors_daily.csv'
# df = pd.read_csv(rf_path, usecols=['Date', 'RF'])
# df = df.rename(columns={
#     'RF': 'RF Rate'
# })
# df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.date
# df['RF Rate'] = df['RF Rate'].astype(float)
# df.to_csv(rf_path, index=False)