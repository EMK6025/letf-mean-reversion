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

base = Portfolio.from_holding(
    close      = spxt,
    freq       = '1D'
)

params = [param0, param1]

pfs = backtest.run(params, "1989-12-31", "2020-12-31")

benchmark_return = base.returns()

# Omega ratio compares average gains over average losses
def omega_ratio(returns, benchmark):
    excess_returns = returns.sub(benchmark, axis=0)
    gains = excess_returns.clip(lower=0)
    losses = excess_returns.clip(upper=0)
    mean_gains  = gains.mean(axis=0)
    mean_losses  = losses.mean(axis=0).abs()
    return mean_gains / mean_losses.replace(0, np.nan)
    
returns = pfs.returns()
sortino = returns.vbt.returns.sortino_ratio(required_return=0.000195)
omega = omega_ratio(returns, benchmark_return)
relative_drawdowns = base.max_drawdown() / pfs.max_drawdown()
alpha = pfs.annualized_return() - base.annualized_return()

fitness_value = fitness(sortino, omega, relative_drawdowns, alpha)

print(fitness_value)

# etf1_value = base.value()
# strat1 = pfs.iloc[0].value()
# strat2 = pfs.iloc[1].value()

# combined = pd.DataFrame({
#     '1x': etf1_value,
#     'strat1': strat1,
#     'strat2': strat2,
# })

# combined.plot(figsize=(12, 6), title='Simulated LETF Performance')
# plt.ylabel('Portfolio Value')
# plt.grid(True)
# plt.show()
