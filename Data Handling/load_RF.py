import pandas as pd
from sql_functions import update_sql_table
from engine import connect

## Connect to SQL
engine = connect()

df = pd.read_csv('F-F_Research_Data_Factors_daily.CSV', usecols=['Date', 'RF'])

df = df.rename(columns={
    "RF": "RF_Rate"
})

df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d").dt.date
df['RF_Rate'] = df['RF_Rate'].astype(float)

update_sql_table(df, engine, "RF")