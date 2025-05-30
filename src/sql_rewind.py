import pandas as pd
from engine import connect

def main():
    engine = connect()
    df = pd.read_sql("SELECT * FROM data", engine)
    df.to_sql("test_data", engine, if_exists="replace", index=False)

    print("test_data entries have been reset.\n")

if __name__ == "__main__":
    main()