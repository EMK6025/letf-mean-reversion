import pandas as pd
from sql_functions import update_sql_table
from engine import create_engine

def main():
    ## connect to SQL
    engine = create_engine()

    df = pd.read_csv('UPRO-historical_nav.csv', usecols=['Date', 'NAV Change (%)'])

    df = df.rename(columns={ # rename
        "NAV Change (%)": "3x LETF Change"
    })

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=False).dt.date
    df['3x LETF Change'] = df['3x LETF Change'].astype(float)

    update_sql_table(df, engine, "UPRO")

if __name__ == "__main__":
    main()