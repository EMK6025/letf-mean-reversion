import pandas as pd
from sql_functions import update_sql_table
from engine import create_engine
from pathlib import Path

def main():
    # connect to SQL
    engine = create_engine()
    proj_path = Path(__file__).resolve().parent.parent 
    
    csv_path = proj_path / 'combined-data.csv'
    df = pd.read_csv(csv_path, usecols=['Date', 'RF Rate'])

    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').dt.date
    df['RF Rate'] = df['RF Rate'].astype(float)

    update_sql_table(df, engine, 'RF')

