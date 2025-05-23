import pandas as pd
from pandas import Series
from engine import create_engine, connect
from sql_functions import update_sql_table

def theoretical_return(p_series: Series, start_val): 
    # actual change
    factors = 1 + p_series / 100
    factors.iloc[0] = start_val
    return factors.cumprod()

def main():
    engine = create_engine()
    df = connect(engine, "test_data")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    
    start_val = df["SPX Close"].iloc[0]
    df["2x LETF"] = theoretical_return(df["2x LETF Change"], start_val)
    df["3x LETF"] = theoretical_return(df["3x LETF Change"], start_val)
    df["4x LETF"] = theoretical_return(df["4x LETF Change"], start_val)
    
    df.reset_index(inplace=True)
    update_sql_table(df, engine, "LETF")

if __name__ == "__main__":
    main()  