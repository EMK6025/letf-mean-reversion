import pandas as pd
from sql_functions import update_sql_table
from engine import connect

def main():
    ## Connect to SQL
    engine = connect()

    df = pd.read_csv('SSO-historical_nav.csv', usecols=['Date', 'NAV Change (%)'])

    df = df.rename(columns={
        "NAV Change (%)": "2x LETF Change"
    })

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=False).dt.date
    df['2x LETF Change'] = df['2x LETF Change'].astype(float)

    update_sql_table(df, engine, "SSO")

if __name__ == "__main__":
    main()