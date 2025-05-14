import pandas as pd
from engine import connect
import vectorbt as vbt
from sql_functions import update_sql_table

def main():
    engine = connect()
    df = pd.read_sql("SELECT * FROM test_data", engine)
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    series = pd.to_numeric(df["SPX Close"], errors="coerce").astype(float)

    for i in range(3,21):
        rsi = vbt.RSI.run(series, window=i, ewm=True)
        df[f"{i} Day RSI"] = rsi.rsi
        print(f"Finished processing {i} Day RSI\n")

    with pd.option_context('display.max_columns', None):
        view = df.tail(20)
        print(view)

    
if __name__ == "__main__":
    main()