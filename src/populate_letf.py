import pandas as pd
from engine import create_engine, connectS
from sql_functions import update_sql_table

ERROR = [0, 0, -0.01029458456440809, -0.010536010464299613, -0.01077743636]

def theoretical_return(df, X):
    return df["SPXTR Change"]*X - df["RF Rate"]*(X-1) - ERROR[X]

def main():
    engine = create_engine()
    df = connect(engine, "test_data")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    
    df["2x LETF Change"] = theoretical_return(df, 2)
    df["3x LETF Change"] = theoretical_return(df, 3)
    df["4x LETF Change"] = theoretical_return(df, 4)
    
    df.reset_index(inplace=True)
    update_sql_table(df, engine, "LETF")

if __name__ == "__main__":
    main()