import pandas as pd
from engine import create_engine, connect
import numpy as np
import vectorbt as vbt
from dataclasses import dataclass, field

@dataclass
class Params:
    window: int
    entry: float
    exit: float
    position_sizing: np.ndarray = field(default_factory=lambda: np.full(10, 0.1))


def evaluate_params(params: Params, price: pd.Series, rsi: pd.Series):
    
    entries = rsi.rsi_crossed_above(params.entry, wait=params.window)
    exits = rsi.rsi_crossed_below(params.exit, wait=params.window)
    size = params.position_sizing
    
    pf = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        size=size,
        size_type='percent',
    )
    return pf

def main():
    engine = create_engine()
    df = connect(engine, "test_data")   
    

if __name__ == "__main__":
    main()