import pandas as pd
from pandas import Series
from engine import create_engine, connect
from sql_functions import update_sql_table

ERROR = [0, 0, -0.01029458456440809, -0.010536010464299613, -0.01077743636]

def theoretical_change(df, X):
    # percent change
    # although UPRO and SSO aim to achieve leveraged SPX returns, they use total swaps
    # thus, they are essentially exposed to SPXTR before fees and tracking errors
    return df["SPXTR Change"]*X - df["RF Rate"]*(X-1) - ERROR[X]

def main():
    engine = create_engine()
    df = connect(engine, "test_data")
    
    df["2x LETF Change"] = theoretical_change(df, 2)
    df["3x LETF Change"] = theoretical_change(df, 3)
    df["4x LETF Change"] = theoretical_change(df, 4)
    
    df.reset_index(inplace=True)
    update_sql_table(df, engine, "LETF Change")
    

if __name__ == "__main__":
    main()  