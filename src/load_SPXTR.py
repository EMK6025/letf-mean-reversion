import pandas as pd
from sql_functions import update_sql_table
from engine import create_engine

def main():
    ## connect to SQL
    engine = create_engine()

    df = pd.read_csv('SPXTR-data.csv', usecols=['Date', 'Close'])

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=False).dt.date
    df['Close'] = df['Close'].astype(float)

    df = df.sort_values("Date")
    df["SPXTR Change"] = df["Close"].pct_change() * 100
    
    df = df.drop(columns="Close")
    
    update_sql_table(df, engine, "SPXTR")

if __name__ == "__main__":
    main()