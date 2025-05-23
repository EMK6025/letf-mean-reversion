import pandas as pd
from sqlalchemy import text, MetaData, Table, Date, Float, Column
import numpy as np

def update_sql_table(df, engine, input_name, table_name="test_data"):
    '''
    desc: updates entries in sql table with dataframe input
    in: dataframe df, engine connection, and optional table name defaulting to test_data
    out: void, simply updates the table
    '''
    df_db = pd.read_sql(f"SELECT * from {table_name}", engine)

    df_db.set_index("Date", inplace=True)
    df.set_index("Date", inplace=True)
    df_db.update(df)

    # add new rows that aren’t already present
    temp = df[~df.index.isin(df_db.index)].copy()
    for col in df_db.columns:
        if col not in temp.columns:
            temp[col] = np.nan
    temp = temp[df_db.columns]

    if df_db.empty:
        df_combined = temp
    elif temp.empty:
        df_combined = df_db
    else:
        df_combined = pd.concat([df_db, temp])

    for col in df_combined.columns:
        if col != "Date":
            df_combined[col] = pd.to_numeric(df_combined[col], errors="coerce").astype(float)

    # rewrite back into sql
    df_combined.reset_index(inplace=True)
    df_combined.to_sql(f"{table_name}", engine, if_exists="replace", index=False)

    with engine.connect() as conn:
        conn.execute(text(f'ALTER TABLE {table_name} ADD PRIMARY KEY ("Date")'))
    
    print(f"{table_name} successfully updated with {input_name} data.\n")

def clean(engine, table_name="test_data"):
    '''
    desc: cleans data by removing unnecessary entries
    in: engine connection and optional table name defaulting to test_data
    out: void, simply updates the table
    '''
    with engine.connect() as conn:
        conn.execute(text(f"""
            DELETE FROM {table_name}
            WHERE "Date" < '1990-11-30' OR "Date" > '2024-12-31'
        """))
        conn.commit()
    
    print(f"entries in {table_name} before 1990-11-30 and after 2024-12-31 have been removed.\n")

def reset(engine):
    '''
    desc: resets both tables
    in: engine connection
    out: void, simply updates the table
    '''
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS data"))
        conn.execute(text("DROP TABLE IF EXISTS test_data"))
        conn.commit()
        
        
    metadata = MetaData()

    data = Table( 
        "data", metadata,
        Column("Date", Date, primary_key=True),
        Column("RF Rate", Float),
        Column("SPXTR Change", Float),
        Column("SPX Close", Float),
        Column("2x LETF", Float),
        Column("3x LETF", Float),
        Column("4x LETF", Float),
        Column("2x LETF Change", Float),
        Column("3x LETF Change", Float),
        Column("4x LETF Change", Float)
        # Column("3 day RSI", Float),
        # Column("4 day RSI", Float),
        # Column("5 day RSI", Float),
        # Column("6 day RSI", Float),
        # Column("7 day RSI", Float),
        # Column("8 day RSI", Float),
        # Column("9 day RSI", Float),
        # Column("10 day RSI", Float),
        # Column("11 day RSI", Float),
        # Column("12 day RSI", Float),
        # Column("13 day RSI", Float),
        # Column("14 day RSI", Float),
        # Column("15 day RSI", Float),
        # Column("16 day RSI", Float),
        # Column("17 day RSI", Float),
        # Column("18 day RSI", Float),
        # Column("19 day RSI", Float),
        # Column("20 day RSI", Float)
    ) 

    test_data = Table( 
        "test_data", metadata,
        Column("Date", Date, primary_key=True),
        Column("RF Rate", Float),
        Column("SPXTR Change", Float),
        Column("SPX Close", Float),
        Column("2x LETF", Float),
        Column("3x LETF", Float),
        Column("4x LETF", Float),
        Column("2x LETF Change", Float),
        Column("3x LETF Change", Float),
        Column("4x LETF Change", Float)
        # Column("3 Day RSI", Float),
        # Column("4 Day RSI", Float),
        # Column("5 Day RSI", Float),
        # Column("6 Day RSI", Float),
        # Column("7 Day RSI", Float),
        # Column("8 Day RSI", Float),
        # Column("9 Day RSI", Float),
        # Column("10 Day RSI", Float),
        # Column("11 Day RSI", Float),
        # Column("12 Day RSI", Float),
        # Column("13 Day RSI", Float),
        # Column("14 Day RSI", Float),
        # Column("15 Day RSI", Float),
        # Column("16 Day RSI", Float),
        # Column("17 Day RSI", Float),
        # Column("18 Day RSI", Float),
        # Column("19 Day RSI", Float),
        # Column("20 Day RSI", Float)
    ) 

    metadata.create_all(engine)
    print(f"data and test_data have been reset to empty structures.\n")

def show(engine, table_name="test_data"):
    df = pd.read_sql(f"SELECT * from {table_name}", engine)

    with pd.option_context('display.max_columns', None):
        print("\n>>> Random 10-row sample:\n")
        print(df.sample(10))

        print("\n>>> First 5 rows:\n")
        print(df.head())

        print("\n>>> Last 5 rows:\n")
        print(df.tail())

        print("\n>>> Summary statistics:\n")
        print(df.describe())