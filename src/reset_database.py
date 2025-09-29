from sql_functions import reset, clean, show
import pandas as pd
from engine import create_engine, connect_time_series

def repopulate_test_data():
    import load_RF, load_ETFs, load_LETFs, load_LETF_change
    engine = create_engine()
    reset(engine)
    # load borrowing costs
    load_RF.main()
    # load SPX and SPXTR
    load_ETFs.main()

    # load estimated daily change for LETFs 
    # (also replaces estimates with actual SSO and UPRO values where available)
    load_LETF_change.main()
    
    # cut off anything that won't be used
    clean(engine)
    # find actual stock values from daily % changes
    load_LETFs.main()

if __name__ == '__main__':
    repopulate_test_data()
    engine = create_engine()
    df = connect_time_series(engine, 'test_data')
    with pd.option_context('display.max_columns', None):
        print('\n>>> Random 10-row sample:\n')
        print(df.sample(10))