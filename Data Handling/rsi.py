import pandas as pd
from engine import connect
import vectorbt as vbt
from sql_functions import update_sql_table

engine = connect()
df = pd.read_sql("SELECT * FROM test_data", engine)
df.set_index("Date", inplace=True)

for i in range(3,21):
    rsi = vbt.RSI.run(df["SPXTR Change"], window=i)
    df[f"{i} Day RSI"] = rsi.rsi
    print(f"Finished processing {i} Day RSI\n")

with pd.option_context('display.max_columns', None):
    view = df.head(20)
    print(view)