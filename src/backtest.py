import pandas as pd
from engine import create_engine, connect
import numpy as np
import vectorbt as vbt
from dataclasses import dataclass, field
import datetime

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

import numpy as np
import pandas as pd

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
    
    print(f"\nprice: {price.shape}")
    print(f"size: {size.shape}")
    print(f"sell_threshold: {sell_threshold.shape}")
    print(f"prelim_entry: {prelim_entry.shape}")
    print(f"prelim_exit: {prelim_exit.shape}")

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
        cycle_watch_price[new_cycle] = price[t] * (100.0 - sell_threshold[new_cycle]) / 100.0
        # finish
        cur_buy[can_buy] += 1
        size_array[t, can_buy] = size[idx_n[can_buy], cur_buy[can_buy]]

        # 4) RSI-exit signals that aren’t high enough → start watching
        valid_exit = prelim_exit[t] & (cur_buy != 0) & (price[t] >= cycle_watch_price)
        invalid_exit = prelim_exit[t] & ~valid_exit
        
        cur_buy[valid_exit] = 0 # reset cycle
        size_array[t, valid_exit] = size[idx_n[valid_sell], 0]
        watching[valid_exit] = False
        cycle_watch_price[valid_exit] = -1
        
        watching[invalid_exit] = True

    return size_array

START = "1989-12-31"

def main():
    engine = create_engine()
    df = connect(engine, "test_data")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    price = df["SPX Close"]
    letf = df["3x LETF"]
    
    param1 = Params(
        window=5,
        entry=20,
        exit=70,
        sell_threshold=10,
        position_sizing = [.2, .6, .8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    
    param2 = Params(
        window=5,
        entry=20,
        exit=70,
        sell_threshold=10,
        position_sizing = [0, .3, .6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    
    params = [param1, param2]
    window_range = np.arange(3, 21)
    rsi = vbt.RSI.run(price, window=window_range, ewm=True)
    windows = np.array([param.window for param in params])
    idx = [(w, True) for w in windows]
    rsi_data = rsi.rsi.loc[START:, idx]
    
    entries = np.array([param.entry for param in params])
    exits = np.array([param.exit for param in params])
    position_sizes = np.vstack([param.position_sizing for param in params]) 
    position_sizes = np.transpose(position_sizes, (1, 0))
    sell_thresholds = np.array([[param.sell_threshold] for param in params])
    sell_thresholds = np.transpose(sell_thresholds, (1, 0))
    
    letf = letf.loc[START:]
    price = price.loc[START:]
    
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
    
    print(f"Shape of letf: {letf.shape}")
    print(f"Shape of prelim_entries: {prelim_entries.shape}")
    print(f"Shape of prelim_exits: {prelim_exits.shape}")
    print(f"Shape of position_sizes: {position_sizes.shape}")
    print(f"Shape of sell_thresholds: {sell_thresholds.shape}")

    # indicator = vbt.IndicatorFactory(
    #     input_names  = ['price', 'prelim_entry', 'prelim_exit'],
    #     param_names  = ['size', 'sell_threshold'],
    #     output_names = ['size_array']
    # ).from_apply_func(
    #     apply
    # )
    indicator = vbt.IndicatorFactory(
        input_names  = ['price', 'prelim_entry', 'prelim_exit', 'size', 'sell_threshold'],
        output_names = ['size_array']
    ).from_apply_func(
        apply
    )

    ind = indicator.run(
        price           = letf,
        prelim_entry    = prelim_entries,
        prelim_exit     = prelim_exits,
        size            = position_sizes,
        sell_threshold  = sell_thresholds
    )
    
    # orders = target_pct.copy()
    # orders[target_pct.eq(target_pct.shift())] = np.nan
    # orders.iloc[0] = target_pct.iloc[0]
    
    target_pct = ind.size_array
    
    orders = pd.DataFrame(target_pct,
                    index=[f"param_{i}" for i in range(len(params))],
                    columns=price.index)
    
    mask = orders.eq(orders.shift(axis=1))
    orders = orders.mask(mask)
    orders.iloc[:, 0] = target_pct[:, 0] # ensure first‐bar orders
    
    pf = vbt.Portfolio.from_orders(
        close      = letf,
        size       = orders,
        size_type  = 'targetpercent',
        freq       = '1D'
    )
    
    print(pf.stats())
    # fig = pf.plot()
    # fig.show()
    # exit_trades = pf.exit_trades
    # et_df = exit_trades.records_readable  
    # trade_85 = et_df.loc[85] 
    # print(trade_85)

if __name__ == "__main__":
    main()