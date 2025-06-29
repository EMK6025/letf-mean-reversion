# src/wfo.py
import go
from engine import create_engine, connect
from vectorbt import Portfolio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from engine import create_engine, connect
import vectorbt as vbt
import warnings
import random
import backtest
import math
from backtest import Params
from deap import tools

warnings.filterwarnings("ignore", category=FutureWarning, module='vectorbt')
warnings.filterwarnings("ignore", category=RuntimeWarning)

vbt.settings.array_wrapper['freq'] = '1D'

seed = 123
random.seed(seed)
np.random.seed(seed)

def window_backtest(start_date, end_date, max_time_minutes=10, stall_generations=10, pop_size=500):
    start_time = time.time()
    max_time_seconds = max_time_minutes * 60
    
    hypervolume_history = []
    stall_counter = 0
    generation = 0
    best_hypervolume = 0
    
    print("Generating initial population...")
    population = go.create_initial_population(pop_size=pop_size)
    
    print("Calculating initial Pareto front...")
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    
    ref_point = [-1.0, -1.0, 2.0, -0.5]  # sortino, sharpe, rel_dd, alpha
    
    def calculate_hypervolume(front):
        if not front:
            return 0
        volume = 0
        for ind in front:
            if ind.fitness.values[0] > -1000:
                vol = 1
                for i, (val, ref) in enumerate(zip(ind.fitness.values, ref_point)):
                    if i == 2:
                        vol *= max(0, ref - val)
                    else:
                        vol *= max(0, val - ref)
                volume += vol
        return volume
    
    print("Calculating initial hypervolume...")
    current_hypervolume = calculate_hypervolume(pareto_front)
    best_hypervolume = current_hypervolume
    hypervolume_history.append(current_hypervolume)
    
    print(f"\nGeneration {generation}: Pareto front size = {len(pareto_front)}, Hypervolume = {current_hypervolume:.4f}")
    
    while True:
        generation += 1
        elapsed_time = time.time() - start_time
        
        if elapsed_time > max_time_seconds:
            print(f"Time limit reached ({max_time_minutes} minutes)")
            break
        
        print(f"\nGeneration {generation} (Elapsed: {elapsed_time/60:.1f}min)...")
        
        population = go.create_next_generation(population, start_date=start_date, end_date=end_date)
        
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        current_hypervolume = calculate_hypervolume(pareto_front)
        
        if current_hypervolume > best_hypervolume:
            improvement = current_hypervolume - best_hypervolume
            stall_counter = 0
            best_hypervolume = current_hypervolume
            print(f"NEW BEST! Hypervolume = {current_hypervolume:.4f} (+{improvement:.4f}), Pareto size = {len(pareto_front)}")
        else:
            stall_counter += 1
            print(f"Hypervolume = {current_hypervolume:.4f} (Stall: {stall_counter}/{stall_generations}), Pareto size = {len(pareto_front)}")
        
        hypervolume_history.append(current_hypervolume)
        
        if stall_counter >= stall_generations:
            print(f"Optimization stalled for {stall_generations} generations")
            break
    
    total_time = time.time() - start_time
    print(f"\nTotal optimization time: {total_time/60:.2f} minutes")
    print(f"Final generation: {generation}")
    print(f"Final Pareto front size: {len(pareto_front)}")
    
    return pareto_front, generation, hypervolume_history

def analyze_pareto_front(pareto_front, start_date):
    """
    Analyze and display the Pareto front strategies with enhanced visualization.
    """
    if not pareto_front:
        print("No valid strategies to analyze!")
        return
    
    print("\n" + "="*80)
    print("PARETO FRONT ANALYSIS")
    print("="*80)
    
    print(f"\nPareto Front Size: {len(pareto_front)} strategies")
        
    best_sortino = max(pareto_front, key=lambda x: x.fitness.values[0] if x.fitness.values[0] > -1000 else -float('inf'))
    print(f"\nBest Sortino: {best_sortino.fitness.values[0]:.3f}")
    print(f"   Full fitness: S={best_sortino.fitness.values[0]:.3f}, Sh={best_sortino.fitness.values[1]:.3f}, "
          f"RD={best_sortino.fitness.values[2]:.3f}, a={best_sortino.fitness.values[3]:.3f}")
    
    best_sharpe = max(pareto_front, key=lambda x: x.fitness.values[1] if x.fitness.values[0] > -1000 else -float('inf'))
    print(f"\nBest Sharpe: {best_sharpe.fitness.values[1]:.3f}")
    print(f"   Full fitness: S={best_sharpe.fitness.values[0]:.3f}, Sh={best_sharpe.fitness.values[1]:.3f}, "
          f"RD={best_sharpe.fitness.values[2]:.3f}, a={best_sharpe.fitness.values[3]:.3f}")
    
    best_rel_dd = min(pareto_front, key=lambda x: x.fitness.values[2] if x.fitness.values[0] > -1000 else float('inf'))
    print(f"\nBest Relative Drawdown: {best_rel_dd.fitness.values[2]:.3f}")
    print(f"   Full fitness: S={best_rel_dd.fitness.values[0]:.3f}, Sh={best_rel_dd.fitness.values[1]:.3f}, "
          f"RD={best_rel_dd.fitness.values[2]:.3f}, a={best_rel_dd.fitness.values[3]:.3f}")
    
    best_alpha = max(pareto_front, key=lambda x: x.fitness.values[3] if x.fitness.values[0] > -1000 else -float('inf'))
    print(f"\nBest Alpha: {best_alpha.fitness.values[3]:.3f}")
    print(f"   Full fitness: S={best_alpha.fitness.values[0]:.3f}, Sh={best_alpha.fitness.values[1]:.3f}, "
          f"RD={best_alpha.fitness.values[2]:.3f}, a={best_alpha.fitness.values[3]:.3f}")
    
    def rank_score(ind):
        vals = ind.fitness.values
        if (
            not all(math.isfinite(v) for v in vals) or
            vals[0] <= -1000   or  # sortino 
            vals[1] <= -1000   or  # sharpe 
            vals[2] >=  1000   or  # rel_dd 
            vals[3] <= -1000      # alpha 
        ):
            return float('inf')
        
        ranks = []
        # Sortino
        ranks.append(sum(1 for x in pareto_front if x.fitness.values[0] > ind.fitness.values[0]))
        # Sharpe
        ranks.append(sum(1 for x in pareto_front if x.fitness.values[1] > ind.fitness.values[1]))
        # Rel DD
        ranks.append(sum(1 for x in pareto_front if x.fitness.values[2] < ind.fitness.values[2]))
        # Alpha
        ranks.append(sum(1 for x in pareto_front if x.fitness.values[3] > ind.fitness.values[3]))
        return sum(ranks) / len(ranks)
    
    balanced_strategy = min(pareto_front, key=rank_score)
    
    print(f"\nMost Balanced Strategy:")
    print(f"   Fitness: S={balanced_strategy.fitness.values[0]:.3f}, Sh={balanced_strategy.fitness.values[1]:.3f}, "
          f"RD={balanced_strategy.fitness.values[2]:.3f}, a={balanced_strategy.fitness.values[3]:.3f}")
    
    window, entry, exit_, sell_threshold, *pos_sizing = balanced_strategy
    
    print(f"\nBalanced Strategy Parameters:")
    print(f"   RSI Window: {window}")
    print(f"   Entry Threshold: {entry}")
    print(f"   Exit Threshold: {exit_}")
    print(f"   Sell Threshold: {sell_threshold}%")
    print(f"   Position Sizing: {[round(x, 3) for x in pos_sizing]}")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Pareto Front Trade-offs (Red = Pareto Optimal)', fontsize=14)
    
    objectives = ['Sortino', 'Sharpe', 'Rel DD', 'Alpha']
    obj_indices = [0, 1, 2, 3]
    
    plot_idx = 0
    for i in range(len(objectives)):
        for j in range(i+1, len(objectives)):
            row = plot_idx // 3
            col = plot_idx % 3
            ax = axes[row, col]
            
            x_vals = [ind.fitness.values[obj_indices[i]] for ind in pareto_front if ind.fitness.values[0] > -1000]
            y_vals = [ind.fitness.values[obj_indices[j]] for ind in pareto_front if ind.fitness.values[0] > -1000]
            
            ax.scatter(x_vals, y_vals, alpha=0.8, color='red', s=50, label='Pareto Front')
            ax.set_xlabel(objectives[i])
            ax.set_ylabel(objectives[j])
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plot_idx += 1
    
    if plot_idx < 6:
        fig.delaxes(axes.flatten()[plot_idx])
    
    plt.tight_layout()

    engine = create_engine()
    df = connect(engine, "test_data")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    spxt = df["SPX Close"][start_date:end_date]
    UPRO = df["3x LETF"][start_date:end_date]

    base = Portfolio.from_holding(
        close      = spxt,
        freq       = '1D'
    )
    
    upro = Portfolio.from_holding(
        close = UPRO, 
        freq = '1D'
    )
    etf1_value = base.value()
    etf3_value = upro.value()
    
    print(f"\nRunning backtest for all {len(pareto_front)} Pareto front strategies...")
    params = []
    for individual in pareto_front:
        window, entry, exit_, sell_threshold, *pos_sizing = individual
        params.append(Params(
            window=window,
            entry=entry,
            exit=exit_,
            sell_threshold=sell_threshold,
            position_sizing=np.array(pos_sizing)
        ))
        
    pfs = backtest.run(params, start_date, end_date)

    plt.figure(figsize=(14, 8))
    
    plt.plot(etf1_value.index, etf1_value.values, 
             label='1x SPX', color='blue', linewidth=2, alpha=0.8)
    plt.plot(etf3_value.index, etf3_value.values, 
             label='3x LETF', color='green', linewidth=2, alpha=0.8)
    
    # Plot all Pareto front strategies in red with varying transparency
    for i, pf in enumerate(pfs):
        alpha = 0.3 + 0.4 * (i / pfs.value().shape[1])
        plt.plot(pf.value().index, pf.value().values, 
                color='red', alpha=alpha, linewidth=1)
    
    plt.plot([], [], color='red', alpha=0.7, linewidth=2, 
             label=f'Pareto Front ({len(pareto_front)} strategies)')
    
    plt.title('All Pareto Front Strategies vs Benchmarks', fontsize=14)
    plt.ylabel('Portfolio Value (Log Scale)')
    plt.xlabel('Date')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    final_values = [pf.value().iloc[-1] for pf in pfs]
    annual_returns = [pf.annualized_return() for pf in pfs]
    max_drawdowns = [pf.max_drawdown() for pf in pfs]
    
    print(f"\nPareto Front Performance Summary:")
    print(f"Final Values - Min: ${min(final_values):,.0f}, Max: ${max(final_values):,.0f}, Median: ${np.median(final_values):,.0f}")
    print(f"Annual Returns - Min: {min(annual_returns):.1%}, Max: {max(annual_returns):.1%}, Median: {np.median(annual_returns):.1%}")
    print(f"Max Drawdowns - Min: {min(max_drawdowns):.1%}, Max: {max(max_drawdowns):.1%}, Median: {np.median(max_drawdowns):.1%}")
    
    return balanced_strategy

def walk_forward_optimization(start_date, end_date, in_sample_months=60, out_sample_months=12, 
                             max_time_minutes=5, stall_generations=5, pop_size=500):

    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    total_months = (end_date_dt.year - start_date_dt.year) * 12 + (end_date_dt.month - start_date_dt.month)
    num_periods = (total_months - in_sample_months) // out_sample_months + 1
    
    wfo_results = []
    all_strategies = []
    all_performances = []
    cumulative_values = [] 
    
    engine = create_engine()
    df = connect(engine, "test_data")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD OPTIMIZATION")
    print(f"In-sample period: {in_sample_months} months, Out-of-sample period: {out_sample_months} months")
    print(f"Total periods: {num_periods}")
    print(f"{'='*80}\n")
    
    current_start = start_date_dt
    current_portfolio_value = None
    
    for period in range(num_periods):
        in_sample_end = current_start + pd.DateOffset(months=in_sample_months)
        out_sample_end = in_sample_end + pd.DateOffset(months=out_sample_months)
        
        if out_sample_end > end_date_dt:
            out_sample_end = end_date_dt
        
        in_sample_start_str = current_start.strftime('%Y-%m-%d')
        in_sample_end_str = in_sample_end.strftime('%Y-%m-%d')
        out_sample_end_str = out_sample_end.strftime('%Y-%m-%d')
        
        print(f"\n{'='*80}")
        print(f"PERIOD {period+1}/{num_periods}")
        print(f"In-sample: {in_sample_start_str} to {in_sample_end_str}")
        print(f"Out-of-sample: {in_sample_end_str} to {out_sample_end_str}")
        print(f"{'='*80}\n")
        
        pareto_front, generations, hypervolume_history = window_backtest(
            start_date=in_sample_start_str,
            end_date=in_sample_end_str,
            max_time_minutes=max_time_minutes,
            stall_generations=stall_generations,
            pop_size=pop_size
        )
        
        def rank_score(ind):
            vals = ind.fitness.values
            if (
                not all(math.isfinite(v) for v in vals) or
                vals[0] <= -1000   or  # sortino 
                vals[1] <= -1000   or  # sharpe 
                vals[2] >=  1000   or  # rel_dd 
                vals[3] <= -1000      # alpha 
            ):
                return float('inf')
            
            ranks = []
            # Sortino
            ranks.append(sum(1 for x in pareto_front if x.fitness.values[0] > ind.fitness.values[0]))
            # Sharpe
            ranks.append(3*sum(1 for x in pareto_front if x.fitness.values[1] > ind.fitness.values[1]))
            # Rel DD
            ranks.append(sum(1 for x in pareto_front if x.fitness.values[2] < ind.fitness.values[2]))
            # Alpha
            ranks.append(5*sum(1 for x in pareto_front if x.fitness.values[3] > ind.fitness.values[3]))
            return sum(ranks) / len(ranks)
        
        best_strategy = min(pareto_front, key=rank_score)
        window, entry, exit_, sell_threshold, *pos_sizing = best_strategy
        
        print(f"\nBest strategy for period {period+1}:")
        print(f"   RSI Window: {window}")
        print(f"   Entry Threshold: {entry}")
        print(f"   Exit Threshold: {exit_}")
        print(f"   Sell Threshold: {sell_threshold}%")
        print(f"   Position Sizing: {[round(x, 3) for x in pos_sizing]}")
        print(f"   Fitness: S={best_strategy.fitness.values[0]:.3f}, Sh={best_strategy.fitness.values[1]:.3f}, "
              f"RD={best_strategy.fitness.values[2]:.3f}, a={best_strategy.fitness.values[3]:.3f}")
        
        # Run backtest on out-of-sample period
        params = [Params(
            window=window,
            entry=entry,
            exit=exit_,
            sell_threshold=sell_threshold,
            position_sizing=np.array(pos_sizing)
        )]
        
        out_sample_pfs = backtest.run(params, in_sample_end_str, out_sample_end_str)
        
        pf = out_sample_pfs[0]
        
        if period == 0:
            in_sample_end_dt = pd.to_datetime(in_sample_end_str)
            spx_after_date = df["SPX Close"][in_sample_end_dt:].iloc[:5]
            if not spx_after_date.empty:
                current_portfolio_value = spx_after_date.iloc[0]
                print(f"   Starting portfolio value: ${current_portfolio_value:.2f} (matched to SPX on {spx_after_date.index[0].strftime('%Y-%m-%d')})")
            else:
                current_portfolio_value = 100.0
                print(f"   Starting portfolio value: ${current_portfolio_value:.2f} (default value)")
        
        pf_values = pf.value()
        scaling_factor = current_portfolio_value / pf_values.iloc[0]
        scaled_values = pf_values * scaling_factor
        
        cumulative_values.append(scaled_values)
        
        current_portfolio_value = scaled_values.iloc[-1]
        
        performance = {
            'period': period + 1,
            'in_sample_start': in_sample_start_str,
            'in_sample_end': in_sample_end_str,
            'out_sample_end': out_sample_end_str,
            'strategy': best_strategy,
            'portfolio': pf,
            'scaled_values': scaled_values,
            'final_value': current_portfolio_value,
            'period_return': (scaled_values.iloc[-1] / scaled_values.iloc[0]) - 1,
            'annualized_return': pf.annualized_return(),
            'max_drawdown': pf.max_drawdown(),
            'sharpe_ratio': pf.sharpe_ratio(),
            'sortino_ratio': pf.sortino_ratio()
        }
        
        wfo_results.append(performance)
        all_strategies.append(best_strategy)
        all_performances.append(pf)
        
        print(f"\nOut-of-sample performance:")
        print(f"   Starting Value: ${scaled_values.iloc[0]:.2f}")
        print(f"   Final Value: ${performance['final_value']:.2f}")
        print(f"   Period Return: {performance['period_return']:.2%}")
        print(f"   Annualized Return: {performance['annualized_return']:.2%}")
        print(f"   Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
        print(f"   Sortino Ratio: {performance['sortino_ratio']:.3f}")
        
        if period < num_periods - 1:
            current_start = current_start + pd.DateOffset(months=out_sample_months)
    
    return wfo_results, all_strategies, all_performances, cumulative_values

def analyze_wfo(wfo_results, all_strategies, all_performances, cumulative_values, start_date, end_date):
    """
    Analyze walk-forward optimization results.
    
    Args:
        wfo_results: Results from walk_forward_optimization
        all_strategies: List of best strategies per period
        all_performances: List of portfolio performances per period
        cumulative_values: List of scaled portfolio values per period
        start_date: Start date of entire optimization
        end_date: End date of entire optimization
    """
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD OPTIMIZATION ANALYSIS")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*80}\n")
    
    returns = [result['annualized_return'] for result in wfo_results]
    period_returns = [result['period_return'] for result in wfo_results]
    drawdowns = [result['max_drawdown'] for result in wfo_results]
    sharpes = [result['sharpe_ratio'] for result in wfo_results]
    sortinos = [result['sortino_ratio'] for result in wfo_results]
    
    print(f"Number of periods: {len(wfo_results)}")
    print(f"Average annualized return: {np.mean(returns):.2%}")
    print(f"Average period return: {np.mean(period_returns):.2%}")
    print(f"Average max drawdown: {np.mean(drawdowns):.2%}")
    print(f"Average Sharpe ratio: {np.mean(sharpes):.3f}")
    print(f"Average Sortino ratio: {np.mean(sortinos):.3f}")
    print(f"Win rate (periods with positive returns): {sum(r > 0 for r in period_returns) / len(period_returns):.2%}")
    
    windows = [strategy[0] for strategy in all_strategies]
    entries = [strategy[1] for strategy in all_strategies]
    exits = [strategy[2] for strategy in all_strategies]
    sell_thresholds = [strategy[3] for strategy in all_strategies]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    ax = axes[0]
    engine = create_engine()
    df = connect(engine, "test_data")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    
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
            label='WFO Strategy', color='red', linewidth=2)
    
    for result in wfo_results:
        ax.axvline(pd.to_datetime(result['in_sample_end']), color='gray', linestyle='--', alpha=0.5)
    
    ax.set_title('Walk-Forward Optimization Performance (Continuous Out-of-Sample)', fontsize=14)
    ax.set_ylabel('Portfolio Value')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    ax = axes[1]
    periods = list(range(1, len(wfo_results) + 1))
    
    ax.plot(periods, windows, marker='o', label='RSI Window')
    ax.plot(periods, entries, marker='s', label='Entry Threshold')
    ax.plot(periods, exits, marker='^', label='Exit Threshold')
    ax.plot(periods, sell_thresholds, marker='*', label='Sell Threshold')
    
    ax.set_title('Parameter Evolution Across Periods', fontsize=14)
    ax.set_ylabel('Parameter Value')
    ax.set_xlabel('Period')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[2]
    
    width = 0.2
    x = np.arange(len(periods))
    
    returns_pct = [r * 100 for r in returns]
    period_returns_pct = [r * 100 for r in period_returns]
    drawdowns_pct = [d * 100 for d in drawdowns]
    
    ax.bar(x - 1.5*width, period_returns_pct, width, label='Period Return %', color='green', alpha=0.7)
    ax.bar(x - 0.5*width, [-d for d in drawdowns_pct], width, label='Max DD % (neg)', color='red', alpha=0.7)
    ax.bar(x + 0.5*width, sharpes, width, label='Sharpe', color='blue', alpha=0.7)
    ax.bar(x + 1.5*width, sortinos, width, label='Sortino', color='purple', alpha=0.7)
    
    ax.set_title('Performance Metrics by Period', fontsize=14)
    ax.set_ylabel('Metric Value')
    ax.set_xlabel('Period')
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nStrategy Parameter Stability Analysis:")
    print(f"RSI Window - Mean: {np.mean(windows):.1f}, Std: {np.std(windows):.1f}, CV: {np.std(windows)/np.mean(windows):.2%}")
    print(f"Entry Threshold - Mean: {np.mean(entries):.1f}, Std: {np.std(entries):.1f}, CV: {np.std(entries)/np.mean(entries):.2%}")
    print(f"Exit Threshold - Mean: {np.mean(exits):.1f}, Std: {np.std(exits):.1f}, CV: {np.std(exits)/np.mean(exits):.2%}")
    print(f"Sell Threshold - Mean: {np.mean(sell_thresholds):.1f}, Std: {np.std(sell_thresholds):.1f}, CV: {np.std(sell_thresholds)/np.mean(sell_thresholds):.2%}")
    
    print("\nPeriod-by-Period Performance:")
    print(f"{'Period':<7} {'In-Sample Start':<15} {'Out-Sample End':<15} {'Return':<12} {'Max DD':<12} {'Sharpe':<10} {'Sortino':<10}")
    print(f"{'-'*85}")
    
    for i, result in enumerate(wfo_results):
        return_str = f"{result['period_return']:.2%}"
        dd_str = f"{result['max_drawdown']:.2%}"
        sharpe_str = f"{result['sharpe_ratio']:.2f}"
        sortino_str = f"{result['sortino_ratio']:.2f}"
        
        print(f"{i+1:<7} {result['in_sample_start']:<15} {result['out_sample_end']:<15} "
              f"{return_str:<12} {dd_str:<12} {sharpe_str:<10} {sortino_str:<10}")
    
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
        
        print(f"\nAggregate Out-of-Sample Performance:")
        print(f"Starting Value: ${first_value:.2f}")
        print(f"Final Value: ${last_value:.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Return/Drawdown Ratio: {-annualized_return/max_drawdown:.2f}")

if __name__ == "__main__":
    start_date = "1990-01-01"
    end_date = "2023-12-31"
    
    print("Starting Walk-Forward Optimization...")
    print(f"Full period: {start_date} to {end_date}")
    
    wfo_results, all_strategies, all_performances, cumulative_values = walk_forward_optimization(
        start_date=start_date,
        end_date=end_date,
        in_sample_months=120,
        out_sample_months=24,
        max_time_minutes=5,
        stall_generations=5,
        pop_size=750
    )
    
    analyze_wfo(wfo_results, all_strategies, all_performances, cumulative_values, start_date, end_date)