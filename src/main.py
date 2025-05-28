from sql_functions import reset, clean, show
import pandas as pd
from engine import create_engine, connect

def repopulate_test_data():
    import load_RF, load_SPXTR, load_SPX, load_SSO, load_UPRO, populate_change, populate_letf
    engine = create_engine()
    reset(engine)
    load_RF.main()
    load_SPXTR.main()
    load_SPX.main()
    populate_change.main()
    load_SSO.main()
    load_UPRO.main()
    clean(engine)
    populate_letf.main()
    

repopulate_test_data()

engine = create_engine()
df = connect(engine, "test_data")
with pd.option_context('display.max_columns', None):
    print("\n>>> Random 10-row sample:\n")
    print(df.sample(10))
# show(engine)
