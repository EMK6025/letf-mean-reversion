
import pandas as pd
import backtest
from backtest import Params
from engine import create_engine, connect
from vectorbt import Portfolio
import numpy as np
import matplotlib.pyplot as plt
import vectorbt as vbt
import warnings
from fitness import calc_metrics

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
    window=4,
    entry=13,
    exit=60,
    sell_threshold=70,
    position_sizing = [0.2, 0.61, 0.63, 0.632, 0.698, 0.703, 0.832, 0.832, 1.0, 1.0, 1.0]
)

start = "1989-12-31"
end = "2020-12-31"

engine = create_engine()
df = connect(engine, "test_data")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)
spxt = df["SPX Close"][start:end]
UPRO = df["3x LETF"][start:end]

base = Portfolio.from_holding(
    close      = spxt,
    freq       = '1D'
)

upro = Portfolio.from_holding(
    close = UPRO, 
    freq = '1D'
)

params = [param0, param1]

pfs = backtest.run(params, start, end)

benchmark_return = base.returns()

sortino, sharpe, rel_drawdown, _, alpha, _ = calc_metrics(pfs, base)

print(f"   2422+: S={sortino.iloc[0]:.3f}, Sh={sharpe.iloc[0]:.3f}, "
        f"RD={rel_drawdown.iloc[0]:.3f}, a={alpha.iloc[0]:.3f}")

print(f"   backtested: S={sortino.iloc[1]:.3f}, Sh={sharpe.iloc[1]:.3f}, "
        f"RD={rel_drawdown.iloc[1]:.3f}, a={alpha.iloc[1]:.3f}")

etf1_value = base.value()
etf3_value = upro.value()
strat1 = pfs.iloc[0].value()
strat2 = pfs.iloc[1].value()

combined = pd.DataFrame({
    '1x': etf1_value,
    '3x': etf3_value,
    '2422+': strat1,
    'backtested': strat2
})

combined.plot(figsize=(12, 6), title='Simulated LETF Performance')
plt.ylabel('Portfolio Value')
plt.grid(True)
plt.show()
