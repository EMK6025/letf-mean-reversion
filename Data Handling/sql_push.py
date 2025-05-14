import pandas as pd
from engine import connect

engine = connect()
df = pd.read_sql("SELECT * FROM test_data", engine)
df.to_sql("data", engine, if_exists="replace", index=False)