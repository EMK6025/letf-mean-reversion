import pandas as pd
from sql_functions import update_sql_table
from engine import create_engine
from pathlib import Path

def main():
    # connect to SQL
    engine = create_engine()
    proj_path = Path(__file__).resolve().parent.parent 
    
    SPX_path = proj_path / 'SPX-data.csv'
    SPXTR_path = proj_path / 'SPXTR-data.csv'
    
    spx = pd.read_csv(SPX_path, usecols=['Date', 'Close'])
    spxtr = pd.read_csv(SPXTR_path, usecols=['Date', 'Close'])
    
    # as spx is used as benchmark, import history directly
    spx['Date'] = pd.to_datetime(spx['Date'], dayfirst=False).dt.date
    spx['Close'] = spx['Close'].astype(float)
    spx = spx.sort_values('Date')
    spx = spx.rename(columns={
        'Close': 'SPX Close'
    })
    update_sql_table(spx, engine, 'SPX')

    # as spxtr is used to simulate pre-inception LETF performance, daily change is needed instead
    spxtr['Date'] = pd.to_datetime(spxtr['Date'], dayfirst=False).dt.date
    spxtr['Close'] = spxtr['Close'].astype(float)
    spxtr = spxtr.sort_values('Date')
    spxtr['SPXTR Change'] = spxtr['Close'].pct_change() * 100
    spxtr = spxtr.drop(columns='Close')
    
    update_sql_table(spxtr, engine, 'SPXTR')
    
if __name__ == '__main__':
    main()