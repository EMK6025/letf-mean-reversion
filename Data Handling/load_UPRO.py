import pandas as pd
from sql_functions import update_sql_table
from engine import connect

## Connect to SQL
engine = connect()

df = pd.read_csv('UPRO-historical_nav.csv', usecols=['Date', 'NAV Change (%)'])

df = df.rename(columns={ # rename
    "NAV Change (%)": "3x LETF Change"
})

df["Date"] = pd.to_datetime(df["Date"], dayfirst=False).dt.date
df['3x LETF Change'] = df['3x LETF Change'].astype(float)

update_sql_table(df, engine, "UPRO")