import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vectorbt import Portfolio
from datetime import date
from engine import create_engine, connect, connect_time_series

def analyze_wfo(run_id):
    """
    Analyze walk-forward optimization results with ensemble strategies.
    """
    cumulative_values, backtest_start_date, leverage = rebuild_performance(run_id)
    engine = create_engine()
    df = connect_time_series(engine, "test_data")
    spx = df["SPX Close"][backtest_start_date:]
    
    pfs = Portfolio.from_holding(
        close=cumulative_values,
        size = 1,
        freq='1D'
    )
    
    benchmark = Portfolio.from_holding(
        close=spx, 
        size=1, 
        freq=pfs.wrapper.freq
    )

    returns = pfs.returns()
    benchmark_returns = benchmark.annualized_return()
    sortino = returns.vbt.returns.sortino_ratio().iloc[0]
    sharpe = returns.vbt.returns.sharpe_ratio().iloc[0]
    rel_drawdown = (pfs.max_drawdown() / benchmark.max_drawdown()).iloc[0]
    drawdown = pfs.max_drawdown().iloc[0]
    alpha = (pfs.annualized_return() - benchmark.annualized_return()).iloc[0]
    annual_return = pfs.annualized_return().iloc[0]
    
    print("=== Performance Metrics ===")
    print(f"Annualized Return       : {annual_return:.2%}")
    print(f"Alpha                   : {alpha:.2%}")
    print(f"Sharpe Ratio            : {sharpe:.2f}")
    print(f"Sortino Ratio           : {sortino:.2f}")
    print(f"Max Drawdown            : {drawdown:.2%}")
    print(f"Relative Drawdown (vs benchmark): {rel_drawdown:.2f}×")
    print(f"Benchmark Annualized Return: {benchmark_returns:.2%}")
    
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(spx.index, spx.values, 
            label='SPX', color='red', linewidth=2, alpha=0.7)
    ax.plot(cumulative_values.index, cumulative_values.values, 
            label='WFO Ensemble Strategy', color='green', linewidth=2)
    
    ax.set_title('Walk-Forward Optimization Ensemble Performance (Continuous Out-of-Sample)', fontsize=14)
    ax.set_ylabel('Portfolio Value')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig("wfo_analysis.png", dpi=300)  # Save the figure with high resolution
    plt.show()
    
def rebuild_performance(run_id):
    from wfo import run_ensemble_backtest

    engine = create_engine()
    run = pd.read_sql(f"SELECT * FROM wfo_run WHERE run_id = {run_id} LIMIT 1", engine)
    strategies = connect(engine, "wfo_strategy")
    strategies = strategies[strategies['run_id'] == run_id]
    start_date = pd.to_datetime(run['start_date'].iloc[0]).date()
    end_date = pd.to_datetime(run['end_date'].iloc[0]).date()
    in_sample_months = run['in_sample_months'].iloc[0]
    out_sample_months = run['out_sample_months'].iloc[0]
    leverage = run['leverage'].iloc[0]
    
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    num_periods = (total_months - in_sample_months) // out_sample_months + 1
    
    backtest_start_date = start_date + pd.DateOffset(months=in_sample_months)
    for col in ['pos_sizing', 'fitness_values']:
        strategies[col] = strategies[col].apply(lambda x: [round(v, 2) for v in x])

    df = connect_time_series(engine, "test_data")
    spx = df["SPX Close"][backtest_start_date:]
    spx_after_date = df["SPX Close"][backtest_start_date:].iloc[0]
    UPRO = df["3x LETF"][backtest_start_date:]

    current_portfolio_value = spx_after_date
    cumulative_values = pd.DataFrame()
    
    for period in range(1, num_periods+1):
        period_strategies = strategies[strategies['period_id'] == period]
        period_start_date = backtest_start_date + pd.DateOffset(months=(period-1)*out_sample_months)
        period_end_date = period_start_date + pd.DateOffset(months=out_sample_months)
        
        capital_per_strategy = current_portfolio_value / len(period_strategies)

        period_ensemble = []
        for i in range(0,len(period_strategies)):
            period_ensemble.append([int(period_strategies["window"].iloc[i]), 
                                    int(period_strategies["entry"].iloc[i]), 
                                    int(period_strategies["exit"].iloc[i]), 
                                    int(period_strategies["sell_threshold"].iloc[i])]
                                   + [float(x) for x in period_strategies["pos_sizing"].iloc[i]])
        
        combined_pf, combined_performance, params = run_ensemble_backtest(period_ensemble, period_start_date, period_end_date, capital_per_strategy, leverage)        
        cumulative_values = pd.concat([cumulative_values, combined_performance], axis=0)
        current_portfolio_value = combined_performance.iloc[-1]
        print(f"Period {period}: Start Date {period_start_date}")
        print(f"                 End Value {current_portfolio_value:.2f}")
        
    return cumulative_values, backtest_start_date, leverage

if __name__ == "__main__":
    analyze_wfo(1)