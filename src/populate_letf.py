import pandas as pd
from pandas import Series
from engine import create_engine, connect_time_series
from sql_functions import update_sql_table

def theoretical_return(p_series: Series, start_val): 
    # actual stock prices
    factors = 1 + p_series / 100
    factors.iloc[0] = start_val
    return factors.cumprod()

def main():
    engine = create_engine()
    df = connect_time_series(engine, 'test_data')
    
    start_val = df['SPX Close'].iloc[0]
    df['2x LETF'] = theoretical_return(df['2x LETF Change'], start_val)
    df['3x LETF'] = theoretical_return(df['3x LETF Change'], start_val)
    df['4x LETF'] = theoretical_return(df['4x LETF Change'], start_val)
    
    df.reset_index(inplace=True)
    update_sql_table(df, engine, 'LETF')
