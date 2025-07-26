from sql_functions import reset, clean, show
import pandas as pd
from engine import create_engine, connect_time_series

def repopulate_test_data():
    import load_RF, load_SPXTR, load_SPX, load_SSO, load_UPRO, populate_change, populate_letf
    engine = create_engine()
    reset(engine)
    # load borrowing costs
    load_RF.main()
    # load SPXTR
    load_SPXTR.main()
    # load SPX
    load_SPX.main()
    # load estimated daily change for LETFs
    populate_change.main()
    # replace estimates with actual for available dates
    load_SSO.main()
    load_UPRO.main()
    # cut off anything that won't be used
    clean(engine)
    # find actual stock values from daily % changes
    populate_letf.main()

if __name__ == "__main__":
    repopulate_test_data()
    engine = create_engine()
    df = connect_time_series(engine, "test_data")
    with pd.option_context('display.max_columns', None):
        print("\n>>> Random 10-row sample:\n")
        print(df.sample(10))