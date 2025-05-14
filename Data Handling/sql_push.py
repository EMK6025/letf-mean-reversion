import pandas as pd
from engine import connect

def main():
    engine = connect()
    df = pd.read_sql("SELECT * FROM test_data", engine)
    df.to_sql("data", engine, if_exists="replace", index=False)

    print("test_data entries have been pushed to data.\n")

if __name__ == "__main__":
    main()