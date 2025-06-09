import pandas as pd
from engine import create_engine, connect
import numpy as np
from vectorbt import Portfolio, RSI, IndicatorFactory
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

@dataclass
class Params:
    window: int
    entry: int
    exit: int
    sell_threshold: int #maximum acceptable cash-out loss on first buy per cycle (%)
    position_sizing: np.ndarray = field(default_factory=lambda: np.full(11, 0.0, dtype=float))
    
    def __post_init__(self):
        arr = np.asarray(self.position_sizing, dtype=float)
        if arr.shape != (11,):
            raise ValueError(
                f"position_sizing must have 11 elements (1 base value and 10 entries), not {arr.shape}"
            )
        self.position_sizing = arr

def parse(input: pd.Series, wait: int) -> pd.Series: # cooldown period
    arr = input.values.astype(bool)
    out = np.zeros_like(arr)
    i = 0
    n = len(arr)
    while i < n:
        if arr[i]:
            out[i] = True
            i += wait + 1
        else:
            i += 1
    return pd.Series(out, index=input.index)
            
def apply(price, prelim_entry, prelim_exit, size, sell_threshold):
    '''
    prelim_entry and prelim_exit is just the unfiltered RSI buy and sell signals, and size is just the buy array
    rebalance is the entry signals that uses accumulate=True and size_type='target_percent'
    exit is only used to track P&L
    size_array is target portfolio percentage
    '''
    
    # price: (T, N)
    # size: (N, 11)
    # sell_threshold: (N,)
    # prelim_entry: (T, N)
    # prelim_exit: (T, N)

    T, N = prelim_entry.shape

    size_array = np.full((T, N), 0, dtype=float)
    cur_buy = np.zeros(N, dtype=int)
    cycle_watch_price = np.full(N, -1, dtype=float)
    watching = np.zeros(N, dtype=bool)

    # for indexing into the first dimension of `size`
    idx_n = np.arange(N)

    for t in range(T):
        size_array[t] = size[idx_n, cur_buy]

        # if watching and price[i] >= cycle_watch_price: # valid sell, reset back to size[0]
        valid_sell = watching & (price[t] >= cycle_watch_price)
        # reset cycle for each one that's true
        size_array[t, valid_sell]     = size[idx_n[valid_sell], 0]
        cur_buy[valid_sell]           = 0
        watching[valid_sell]          = False
        cycle_watch_price[valid_sell] = -1
        
        # new entry signals + need to send new buy order
        can_buy = prelim_entry[t] & (cur_buy < 10)
        # fresh cycles need to set a new watch price:
        new_cycle = can_buy & (cur_buy == 0)
        threshold = price[t] * (100.0 - sell_threshold) / 100.0  
        cycle_watch_price[new_cycle] = threshold[new_cycle]

        # finish
        cur_buy[can_buy] += 1
        size_array[t, can_buy] = size[idx_n[can_buy], cur_buy[can_buy]]

        # 4) RSI-exit signals that aren’t high enough → start watching
        has_position = cur_buy > 0
        valid_exit = prelim_exit[t] & has_position & (price[t] >= cycle_watch_price)
        invalid_exit = prelim_exit[t] & ~valid_exit & has_position
        
        cur_buy[valid_exit] = 0 # reset cycle
        size_array[t, valid_exit] = size[idx_n[valid_exit], 0]
        watching[valid_exit] = False
        cycle_watch_price[valid_exit] = -1
        
        watching[invalid_exit] = True

    return size_array

def run(params, start_date, end_date):
    print("starting run")
    engine = create_engine()
    df = connect(engine, "test_data")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    price = df["SPX Close"]
    letf = df["3x LETF"]
    
    window_range = np.arange(3, 21)
    rsi = RSI.run(price, window=window_range, ewm=True)
    windows = np.array([param.window for param in params])
    idx = [(w, True) for w in windows]
    rsi_data = rsi.rsi.loc[start_date:end_date, idx]
        
    entries = np.array([param.entry for param in params])
    exits = np.array([param.exit for param in params])
    position_sizes = np.vstack([param.position_sizing for param in params]) 
    sell_thresholds = np.array([param.sell_threshold for param in params])
    
    letf = letf.loc[start_date:end_date]
    price = price.loc[start_date:end_date]
    
    entry_mask = rsi_data < entries
    exit_mask = rsi_data > exits
    
    prelim_entries = pd.DataFrame(
        {i: parse(entry_mask.iloc[:, i], windows[i]) 
            for i in range(len(params)) }
    )
    
    prelim_exits = pd.DataFrame(
        {i: parse(exit_mask.iloc[:, i],  windows[i]) 
            for i in range(len(params)) }
    )

    indicator = IndicatorFactory(
        input_names = ['price', 'prelim_entry', 'prelim_exit'],
        param_names = ['size', 'sell_threshold'],
        output_names = ['size_array']
    ).from_apply_func(
        apply,
        param_settings={ # vectorize param_names 
            'size': dict(is_array_like=True),
            'sell_threshold': dict(is_array_like=True)
        }
    )
    ind = indicator.run(
        price           = letf,
        prelim_entry    = prelim_entries,
        prelim_exit     = prelim_exits,
        size            = position_sizes,
        sell_threshold  = sell_thresholds
    )
    target_pct = ind.size_array.to_numpy()

    orders = pd.DataFrame(
        target_pct,
        index=price.index,
        columns=[f"param_{i}" for i in range(target_pct.shape[1])]
    )
    mask = orders.eq(orders.shift(axis=0))
    orders_masked = orders.mask(mask)
    orders_masked.iloc[0, :] = orders.iloc[0, :]
    size_changes = orders_masked.to_numpy()
    
    pf = Portfolio.from_orders(
        close      = letf,
        size       = size_changes,
        size_type  = 'targetpercent',
        freq       = '1D'
    )
    return pf