import pandas as pd
from pandas import Series
from engine import create_engine, connect

def theoretical_return(p_series: Series, start_val):
    factors = 1 + p_series / 100
    factors.iloc[0] = start_val
    return factors.cumprod()

def main():
    engine = create_engine()
    df = connect(engine, "test_data")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    start_val = df["SPX Close"].iloc[0]
    letf_2x = theoretical_return(df["2x LETF Change"], 2)
    letf_3x = theoretical_return(df["3x LETF Change"], 2)
    letf_4x = theoretical_return(df["4x LETF Change"], 2)

    
    print(f"3x Error is {error_3x}\n")
    print(f"2x Error is {error_2x}\n")
    
if __name__ == "__main__":
    main()
    
    