import pandas as pd
from engine import connect
import vectorbt as vbt
from sql_functions import update_sql_table

def theoretical_return(df, X):
    return df["SPXTR Change"]*X - df["RF Rate"]*(X-1)

def main():
    engine = connect()
    df = pd.read_sql("SELECT * FROM test_data", engine)
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    
    df["Theoretical 3x LETF"] = theoretical_return(df, 3)
    temp = df["3x LETF Change"].notna() & df["Theoretical 3x LETF"].notna()
    df.loc[temp, "3x Error"] = ( df["3x LETF Change"] - df["Theoretical 3x LETF"] )
    
    df["Theoretical 2x LETF"] = theoretical_return(df, 2)
    temp = df["2x LETF Change"].notna() & df["Theoretical 2x LETF"].notna()
    df.loc[temp, "2x Error"] = ( df["2x LETF Change"] - df["Theoretical 2x LETF"] )

    with pd.option_context('display.max_columns', None):
        view = df[["SPXTR Change", "3x LETF Change", "Theoretical 3x LETF", "3x Error", "2x LETF Change", "Theoretical 2x LETF", "2x Error"]].tail(10)
        print(view)

if __name__ == "__main__":
    main()
    
    