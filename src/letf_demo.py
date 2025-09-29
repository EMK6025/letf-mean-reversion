import pandas as pd
from engine import create_engine, connect_time_series
from vectorbt import Portfolio
import matplotlib.pyplot as plt

START = '2000-01-01'

def main():
    engine = create_engine()
    df = connect_time_series(engine, 'test_data')
    # df = df.loc[START:]
    spxt = df['SPX Close']
    sso = df['2x LETF']
    upro = df['3x LETF']
    letf4x = df['4x LETF']

    base = Portfolio.from_holding(
        close      = spxt,
        freq       = '1D'
    )
    
    letf_2x = Portfolio.from_holding(
        close      = sso, 
        freq       = '1D'
    )
    letf_3x = Portfolio.from_holding(
        close      = upro, 
        freq       = '1D'
    )
    letf_4x = Portfolio.from_holding(
        close      = letf4x, 
        freq       = '1D'
    )
    
    etf1_value = base.value()
    letf2_value = letf_2x.value()
    letf3_value = letf_3x.value()
    letf4_value = letf_4x.value()

    combined = pd.DataFrame({
        '1x': etf1_value,
        '2x': letf2_value,
        '3x': letf3_value,
        '4x': letf4_value,
    })

    combined.plot(figsize=(12, 6), title='Simulated LETF Performance')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.show()
    print(base.stats())