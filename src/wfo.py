import pandas as pd
import backtest
from backtest import Params
from engine import create_engine, connect
from vectorbt import Portfolio
import numpy as np

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

benchmark_return = base.returns
for pf in pfs:
    max_drawdown = pf.stats()['Max Drawdown [%]']
    returns = pf.returns

    # Calculate Sortino Ratio with custom MAR
    excess_returns = returns - benchmark_return
    downside_returns = excess_returns[excess_returns < 0]
    mean_excess_return = excess_returns.mean()
    downside_deviation = downside_returns.std(ddof=1)

    # Compute Sortino Ratio
    sortino = mean_excess_return / downside_deviation if downside_deviation != 0 else np.nan
    print(f"max drawdown: {max_drawdown}")
    print(f"sortino with SPX as benchmark: {sortino}")
