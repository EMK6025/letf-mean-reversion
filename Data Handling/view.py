import pandas as pd
from engine import connect

def main():
    engine = connect()
    df = pd.read_sql("SELECT * FROM data", engine)

    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    with pd.option_context('display.max_columns', None):
        print("\n>>> Random 10-row sample:\n")
        print(df.sample(10))

        print("\n>>> First 5 rows:\n")
        print(df.head())

        print("\n>>> Last 5 rows:\n")
        print(df.tail())

        print("\n>>> Summary statistics:\n")
        print(df.describe())

if __name__ == "__main__":
    main()
