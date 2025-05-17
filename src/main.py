from sql_functions import reset, clean, show
import pandas as pd
from engine import connect

engine = connect()

def repopulate_test_data():
    import load_RF, load_SPXTR, load_SPX, load_SSO, load_UPRO, load_rsi, populate_letf
    reset(engine)
    load_RF.main()
    load_SPXTR.main()
    load_SPX.main()
    load_rsi.main()
    populate_letf.main()
    load_SSO.main()
    load_UPRO.main()

# repopulate_test_data()
# clean(engine)

show(engine)