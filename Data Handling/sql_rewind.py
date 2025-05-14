import pandas as pd
from engine import connect

engine = connect()
df = pd.read_sql("SELECT * FROM data", engine)
df.to_sql("test_data", engine, if_exists="replace", index=False)