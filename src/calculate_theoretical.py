import pandas as pd
from engine import create_engine, connect

def theoretical_return(df, X):
    return df["SPXTR Change"]*X - df["RF Rate"]*(X-1)

def main():
    engine = create_engine()
    df = connect(engine, "test_data")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    
    df["Theoretical 3x LETF"] = theoretical_return(df, 3)
    temp = df["3x LETF Change"].notna() & df["Theoretical 3x LETF"].notna()
    df.loc[temp, "3x Error"] = ( df["3x LETF Change"] - df["Theoretical 3x LETF"] )
    error_3x = df.loc[temp, "3x Error"].mean()
    
    df["Theoretical 2x LETF"] = theoretical_return(df, 2)
    temp = df["2x LETF Change"].notna() & df["Theoretical 2x LETF"].notna()
    df.loc[temp, "2x Error"] = ( df["2x LETF Change"] - df["Theoretical 2x LETF"] )
    error_2x = df.loc[temp, "2x Error"].mean()
    
    print(f"3x Error is {error_3x}\n")
    print(f"2x Error is {error_2x}\n")
    
if __name__ == "__main__":
    main()
    
    