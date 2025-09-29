import pandas as pd
from engine import create_engine, connect_time_series

def main():
    engine = create_engine()
    df = connect_time_series(engine, 'test_data')
    df.to_sql('data', engine, if_exists='replace', index=False)

    print('test_data entries have been pushed to data.\n')

if __name__ == '__main__':
    main()