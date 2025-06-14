
import pandas as pd
import backtest
from backtest import Params
from engine import create_engine, connect
from vectorbt import Portfolio
import numpy as np
import matplotlib.pyplot as plt
import vectorbt as vbt
import warnings
from go import fitness

warnings.filterwarnings("ignore", category=FutureWarning, module='vectorbt')
vbt.settings.array_wrapper['freq'] = '1D'

param0 = Params(
    window=5,
    entry=20,
    exit=70,
    sell_threshold=10,
    position_sizing = [.2, .6, .8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)

param1 = Params(
    window=5,
    entry=30,
    exit=80,
    sell_threshold=10,
    position_sizing = [.2, .4, .6, .8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)

engine = create_engine()
df = connect(engine, "test_data")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)
spxt = df["SPX Close"]
UPRO = df["3x LETF"]

base = Portfolio.from_holding(
    close      = spxt,
    freq       = '1D'
)

upro = Portfolio.from_holding(
    close = UPRO, 
    freq = '1D'
)

params = [param0]

pfs = backtest.run(params, "1989-12-31", "2024-12-31")

benchmark_return = base.returns()

etf1_value = base.value()
etf3_value = upro.value()
strat1 = pfs.iloc[0].value()

combined = pd.DataFrame({
    '1x': etf1_value,
    '3x': etf3_value,
    '2422+': strat1,
})

combined.plot(figsize=(12, 6), title='Simulated LETF Performance')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.show()
