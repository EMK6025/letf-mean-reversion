import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vectorbt import Portfolio

from engine import create_engine, connect_time_series

def analyze_wfo(wfo_results, all_ensemble_strategies, cumulative_values, start_date, end_date):
    """
    Analyze walk-forward optimization results with ensemble strategies.
    """
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD OPTIMIZATION ENSEMBLE ANALYSIS")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*80}\n")
    
    returns = [result['annualized_return'] for result in wfo_results]
    period_returns = [result['period_return'] for result in wfo_results]
    drawdowns = [result['max_drawdown'] for result in wfo_results]
    sharpes = [result['sharpe_ratio'] for result in wfo_results]
    sortinos = [result['sortino_ratio'] for result in wfo_results]
    n_strategies_per_period = [result['n_strategies'] for result in wfo_results]
    
    print(f"Number of periods: {len(wfo_results)}")
    print(f"Average strategies per period: {np.mean(n_strategies_per_period):.1f}")
    print(f"Average annualized return: {np.mean(returns):.2%}")
    print(f"Average period return: {np.mean(period_returns):.2%}")
    print(f"Average max drawdown: {np.mean(drawdowns):.2%}")
    print(f"Average Sharpe ratio: {np.mean(sharpes):.3f}")
    print(f"Average Sortino ratio: {np.mean(sortinos):.3f}")
    print(f"Win rate (periods with positive returns): {sum(r > 0 for r in period_returns) / len(period_returns):.2%}")
    
    # analyze strategy diversity across periods
    all_windows = []
    all_entries = []
    all_exits = []
    all_sell_thresholds = []
    
    for ensemble in all_ensemble_strategies:
        for strategy in ensemble:
            window, entry, exit_, sell_threshold, *pos_sizing = strategy
            all_windows.append(window)
            all_entries.append(entry)
            all_exits.append(exit_)
            all_sell_thresholds.append(sell_threshold)
    
    print(f"\nStrategy Parameter Diversity (across all {len(all_windows)} strategy instances):")
    print(f"RSI Window - Mean: {np.mean(all_windows):.1f}, Std: {np.std(all_windows):.1f}, Range: {min(all_windows)}-{max(all_windows)}")
    print(f"Entry Threshold - Mean: {np.mean(all_entries):.1f}, Std: {np.std(all_entries):.1f}, Range: {min(all_entries):.1f}-{max(all_entries):.1f}")
    print(f"Exit Threshold - Mean: {np.mean(all_exits):.1f}, Std: {np.std(all_exits):.1f}, Range: {min(all_exits):.1f}-{max(all_exits):.1f}")
    print(f"Sell Threshold - Mean: {np.mean(all_sell_thresholds):.1f}, Std: {np.std(all_sell_thresholds):.1f}, Range: {min(all_sell_thresholds):.1f}-{max(all_sell_thresholds):.1f}")
    
    _, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    ax = axes[0]
    engine = create_engine()
    df = connect_time_series(engine, "test_data")
    
    first_out_sample_start = wfo_results[0]['in_sample_end']
    
    spxt = df["SPX Close"][first_out_sample_start:end_date]
    UPRO = df["3x LETF"][first_out_sample_start:end_date]

    base = Portfolio.from_holding(
        close = spxt,
        freq = '1D'
    )
    
    upro = Portfolio.from_holding(
        close = UPRO, 
        freq = '1D'
    )
    
    combined_performance = pd.concat(cumulative_values)
    combined_performance = combined_performance[~combined_performance.index.duplicated(keep='first')]
    combined_performance = combined_performance.sort_index()
    
    initial_wfo_value = combined_performance.iloc[0]
    base_scaling = initial_wfo_value / base.value().iloc[0]
    upro_scaling = initial_wfo_value / upro.value().iloc[0]
    
    ax.plot(base.value().index, base.value().values * base_scaling, 
            label='1x SPX', color='blue', linewidth=2, alpha=0.7)
    ax.plot(upro.value().index, upro.value().values * upro_scaling, 
            label='3x LETF', color='green', linewidth=2, alpha=0.7)
    ax.plot(combined_performance.index, combined_performance.values, 
            label='WFO Ensemble Strategy', color='red', linewidth=2)
    
    for result in wfo_results:
        ax.axvline(pd.to_datetime(result['in_sample_end']), color='gray', linestyle='--', alpha=0.5)
    
    ax.set_title('Walk-Forward Optimization Ensemble Performance (Continuous Out-of-Sample)', fontsize=14)
    ax.set_ylabel('Portfolio Value')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    # parameter diversity visualization
    ax = axes[1]
    periods = list(range(1, len(wfo_results) + 1))
    
    # calculate average parameters per period
    avg_windows = [np.mean([s[0] for s in ensemble]) for ensemble in all_ensemble_strategies]
    avg_entries = [np.mean([s[1] for s in ensemble]) for ensemble in all_ensemble_strategies]
    avg_exits = [np.mean([s[2] for s in ensemble]) for ensemble in all_ensemble_strategies]
    avg_sell_thresholds = [np.mean([s[3] for s in ensemble]) for ensemble in all_ensemble_strategies]
    
    ax.plot(periods, avg_windows, marker='o', label='Avg RSI Window', linewidth=2)
    ax.plot(periods, avg_entries, marker='s', label='Avg Entry Threshold', linewidth=2)
    ax.plot(periods, avg_exits, marker='^', label='Avg Exit Threshold', linewidth=2)
    ax.plot(periods, avg_sell_thresholds, marker='*', label='Avg Sell Threshold', linewidth=2)
    
    ax.set_title('Average Ensemble Parameters Evolution Across Periods', fontsize=14)
    ax.set_ylabel('Parameter Value')
    ax.set_xlabel('Period')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[2]
    
    width = 0.15
    x = np.arange(len(periods))
    
    returns_pct = [r * 100 for r in returns]
    period_returns_pct = [r * 100 for r in period_returns]
    drawdowns_pct = [d * 100 for d in drawdowns]
    
    ax.bar(x - 2*width, period_returns_pct, width, label='Period Return %', color='green', alpha=0.7)
    ax.bar(x - width, [-d for d in drawdowns_pct], width, label='Max DD % (neg)', color='red', alpha=0.7)
    ax.bar(x, sharpes, width, label='Sharpe', color='blue', alpha=0.7)
    ax.bar(x + width, sortinos, width, label='Sortino', color='purple', alpha=0.7)
    ax.bar(x + 2*width, n_strategies_per_period, width, label='# Strategies', color='orange', alpha=0.7)
    
    ax.set_title('Ensemble Performance Metrics by Period', fontsize=14)
    ax.set_ylabel('Metric Value')
    ax.set_xlabel('Period')
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nPeriod-by-Period Ensemble Performance:")
    print(f"{'Period':<7} {'In-Sample Start':<15} {'Out-Sample End':<15} {'#Strat':<7} {'Return':<12} {'Max DD':<12} {'Sharpe':<10} {'Sortino':<10}")
    print(f"{'-'*100}")
    
    for i, result in enumerate(wfo_results):
        return_str = f"{result['period_return']:.2%}"
        dd_str = f"{result['max_drawdown']:.2%}"
        sharpe_str = f"{result['sharpe_ratio']:.2f}"
        sortino_str = f"{result['sortino_ratio']:.2f}"
        n_strat_str = f"{result['n_strategies']}"
        
        print(f"{i+1:<7} {result['in_sample_start']:<15} {result['out_sample_end']:<15} "
              f"{n_strat_str:<7} {return_str:<12} {dd_str:<12} {sharpe_str:<10} {sortino_str:<10}")
    
    if combined_performance.empty:
        print("\nNo valid combined performance data")
    else:
        first_value = combined_performance.iloc[0]
        last_value = combined_performance.iloc[-1]
        total_return = (last_value / first_value) - 1
        duration_years = (combined_performance.index[-1] - combined_performance.index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1 / duration_years) - 1
        
        combined_performance_series = pd.Series(combined_performance.values, index=combined_performance.index)
        rolling_max = combined_performance_series.cummax()
        drawdown = (combined_performance_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        print(f"\nAggregate Ensemble Out-of-Sample Performance:")
        print(f"Starting Value: ${first_value:.2f}")
        print(f"Final Value: ${last_value:.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Return/Drawdown Ratio: {-annualized_return/max_drawdown:.2f}")