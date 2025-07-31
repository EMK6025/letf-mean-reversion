import pandas as pd
import numpy as np
from engine import create_engine, connect_time_series
import vectorbt as vbt
from sql_functions import update_sql_table


def main():
    engine = create_engine()
    df = connect_time_series(engine, "test_data")
    series = pd.to_numeric(df["SPX Close"], errors="coerce").astype(float)

    windows = np.arange(3, 21)
    rsi = vbt.RSI.run(series, window=windows, ewm=True)
    rsi_df = rsi.rsi
    rsi_df.columns = [f"{w} Day RSI" for w in rsi_df.columns]

    # merge into your existing df
    df.update(rsi_df)


    df.reset_index(inplace=True)  
    update_sql_table(df, engine, "RSI")

