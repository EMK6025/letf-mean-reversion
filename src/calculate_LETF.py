import pandas as pd
from pandas import Series
from engine import create_engine, connect_time_series

def theoretical_return(p_series: Series, start_val):
    factors = 1 + p_series / 100
    factors.iloc[0] = start_val
    return factors.cumprod()

def main():
    engine = create_engine()
    df = connect_time_series(engine, "test_data")
    start_val = df["SPX Close"].iloc[0]
    letf_2x = theoretical_return(df["2x LETF Change"], start_val)
    letf_3x = theoretical_return(df["3x LETF Change"], start_val)
    letf_4x = theoretical_return(df["4x LETF Change"], start_val)
    print(letf_2x)
    print(letf_3x)
    print(letf_4x)
    
if __name__ == "__main__":
    main()