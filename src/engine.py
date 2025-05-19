from sqlalchemy import create_engine
import pandas as pd

def create_engine():
    return create_engine("postgresql+psycopg://postgres:Mhkuu1.1213704313@localhost:5432/letf_data")

def connect(engine, table_name = "test_data"):
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)