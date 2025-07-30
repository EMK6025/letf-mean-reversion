import pandas as pd
import backtest
from backtest import Params
from engine import create_engine, connect_time_series
from vectorbt import Portfolio
import numpy as np
import matplotlib.pyplot as plt
import vectorbt as vbt
import warnings
from fitness import calc_metrics

warnings.filterwarnings("ignore", category=FutureWarning, module='vectorbt')
vbt.settings.array_wrapper['freq'] = '1D'

# def print_summary(strat_name, index, metrics_dict):
#     print(f"   {strat_name}: S={metrics_dict['sortino'].iloc[index]:.3f}, Sh={metrics_dict['sharpe'].iloc[index]:.3f}, "
#         f"RD={metrics_dict['rel_drawdown'].iloc[index]:.3f}, a={metrics_dict['alpha'].iloc[index]:.3f}")

# param0 = Params(
#     window=5,
#     entry=20,
#     exit=70,
#     sell_threshold=10,
#     position_sizing = [.2, .6, .8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
# )

# start = pd.to_datetime("2001-01-01").date()
# end = pd.to_datetime("2024-12-31").date()

# engine = create_engine()
# df = connect_time_series(engine, "test_data")
# spxt = df["SPX Close"][start:]
# UPRO = df["3x LETF"][start:]

# base = Portfolio.from_holding(
#     close      = spxt,
#     freq       = '1D'
# )

# upro = Portfolio.from_holding(
#     close = UPRO, 
#     freq = '1D'
# )

# params = [param0]

# current_portfolio_value = spxt.iloc[0]
# pfs = backtest.run(params, start, end, initial_capital=100)

# benchmark_return = base.returns()

# metrics = calc_metrics(pfs, base, params)

# print_summary("2422+", 0, metrics)

# etf1_value = base.value()
# etf3_value = upro.value()
# strat1 = pfs.iloc[0].value()

# combined = pd.DataFrame({
#     '1x': etf1_value,
#     '3x': etf3_value,
#     '2422+': strat1
# })

# combined.plot(figsize=(12, 6), title='Simulated LETF Performance')
# plt.ylabel('Portfolio Value')
# plt.grid(True)
# plt.show()

engine = create_engine()
run = pd.read_sql(f"SELECT run_id, COUNT(*) AS occurrences FROM wfo_run GROUP BY run_id ORDER BY occurrences DESC;", engine)

print(run)
# import pandas as pd
# import backtest
# from backtest import Params
# from engine import create_engine, connect
# from vectorbt import Portfolio
# import numpy as np


# engine = create_engine()
# runs = connect(engine, "wfo_run")
# period_summaries = connect(engine, "wfo_period_summary")
# strategies = connect(engine, "wfo_strategy")

# strategies = strategies.round(2)
# for col in ['pos_sizing', 'fitness_values']:
#     strategies[col] = strategies[col].apply(lambda x: [round(v, 2) for v in x])

# with pd.option_context('display.max_columns', None, 
#                        'display.max_colwidth', None, 
#                        'display.width', None):
#     print(strategies.head(11))