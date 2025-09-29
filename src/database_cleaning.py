import pandas as pd
from sql_functions import update_sql_table
from engine import create_engine
from pathlib import Path

def main():
    proj_path = Path(__file__).resolve().parent.parent
    spx_path = proj_path / 'SPX-data.csv'
    spxtr_path = proj_path / 'SPXTR-data.csv'
    rf_path = proj_path / 'F-F_Research_Data_Factors_daily.csv'
    
    spx = pd.read_csv(spx_path, usecols=['Date', 'SPX Close'])
    spxtr = pd.read_csv(spxtr_path, usecols=['Date', 'SPXTR Close'])
    rf = pd.read_csv(rf_path, usecols=['Date', 'RF Rate'])
    
    spx['Date'] = pd.to_datetime(spx['Date'], dayfirst=False).dt.date
    spx['SPX Close'] = spx['SPX Close'].astype(float)
    spx = spx.sort_values('Date')
    
    spxtr['Date'] = pd.to_datetime(spxtr['Date'], dayfirst=False).dt.date
    spxtr['SPXTR Close'] = spxtr['SPXTR Close'].astype(float)
    spxtr = spxtr.sort_values('Date')

    rf['Date'] = pd.to_datetime(rf['Date'], dayfirst=False).dt.date
    rf['RF Rate'] = rf['RF Rate'].astype(float)
    rf = rf.sort_values('Date')
    
    spx = spx.set_index('Date')
    spxtr = spxtr.set_index('Date')
    rf = rf.set_index('Date')
    

    combined = spx.join([spxtr, rf], how='inner')
    combined = combined.reset_index()
    combined.to_csv(proj_path / 'combined-data.csv', index=False)

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