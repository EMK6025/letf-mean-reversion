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
from backtest import Params
from deap import tools
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from fitness import FitnessConfig

warnings.filterwarnings("ignore", category=FutureWarning, module='vectorbt')
warnings.filterwarnings("ignore", category=RuntimeWarning)

vbt.settings.array_wrapper['freq'] = '1D'

seed = 69
random.seed(seed)
np.random.seed(seed)

def window_backtest(start_date, end_date, max_time_minutes=10, stall_generations=10, pop_size=500, leverage=3, fitness_config=None):
    
    start_time = time.time()
    max_time_seconds = max_time_minutes * 60
    
    hypervolume_history = []
    stall_counter = 0
    generation = 0
    best_hypervolume = 0
    
    print("Generating initial population...")
    population = go.create_initial_population(pop_size=pop_size, start_date=start_date, end_date=end_date, leverage=leverage)
    
    print("Calculating initial Pareto front...")
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    
    # dynamic reference point based on selected metrics
    ref_point = []
    for metric in fitness_config.selected_metrics:
        params = fitness_config.available_metrics[metric]
        ref_point.append(params[2])
            
    def calculate_hypervolume(front):
        if not front:
            return 0
        volume = 0
        for ind in front:
            if ind.fitness.values[0] > -1000:
                vol = 1
                for i, (val, ref) in enumerate(zip(ind.fitness.values, ref_point)):
                    metric = fitness_config.selected_metrics[i]
                    if fitness_config.is_minimize_metric(metric):
                        vol *= max(0, ref - val)
                    else:
                        vol *= max(0, val - ref)
                volume += vol
        return volume/len(front)
    
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
        
        population = go.create_next_generation(population, start_date=start_date, end_date=end_date, leverage=leverage)
        
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

def select_diverse_strategies(pareto_front, n_strategies=10):
    if len(pareto_front) <= n_strategies:
        print(f"Pareto front has only {len(pareto_front)} strategies, using all of them")
        return pareto_front
    
    # extract strategy parameters for clustering (only RSI window, entry, exit, sell threshold)
    strategy_params = []
    for individual in pareto_front:
        window, entry, exit, sell_threshold, *pos_sizing = individual
        params = [window, entry, exit, sell_threshold] + pos_sizing
        strategy_params.append(params)
    
    strategy_params = np.array(strategy_params)
    
    # standardize the parameters
    scaler = StandardScaler()
    strategy_params_scaled = scaler.fit_transform(strategy_params)
    
    # use K-means clustering to find diverse strategies
    kmeans = KMeans(n_clusters=n_strategies, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(strategy_params_scaled)
    
    # select one strategy from each cluster (closest to cluster center)
    diverse_strategies = []
    for cluster_id in range(n_strategies):
        cluster_mask = cluster_labels == cluster_id
        if not np.any(cluster_mask):
            continue
            
        cluster_strategies = strategy_params_scaled[cluster_mask]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        
        # find the strategy closest to the cluster center
        distances = np.linalg.norm(cluster_strategies - cluster_center, axis=1)
        closest_idx = np.argmin(distances)
        
        # get the original strategy index
        cluster_indices = np.where(cluster_mask)[0]
        original_idx = cluster_indices[closest_idx]
        
        diverse_strategies.append(pareto_front[original_idx])
    
    print(f"Selected {len(diverse_strategies)} diverse strategies from {len(pareto_front)} Pareto front strategies")
    
    return diverse_strategies

def run_ensemble_backtest(strategies, start_date, end_date, initial_capital_per_strategy=10000, leverage=3):
    """
    Run backtest for ensemble of strategies with equal capital allocation.
    """
    print(f"Running ensemble backtest with {len(strategies)} strategies...")
    print(f"Capital per strategy: ${initial_capital_per_strategy:,}")
    
    params = []
    for individual in strategies:
        window, entry, exit_, sell_threshold, *pos_sizing = individual
        params.append(Params(
            window=window,
            entry=entry,
            exit=exit_,
            sell_threshold=sell_threshold,
            position_sizing=np.array(pos_sizing)
        ))
    
    # run backtests for all strategies
    portfolios = backtest.run(params, start_date, end_date, initial_capital_per_strategy, leverage=leverage)
    
    performance = portfolios.value().ffill().sum(axis=1)
    return performance, portfolios

def walk_forward_optimization(start_date, end_date, in_sample_months=60, out_sample_months=12, 
                             max_time_minutes=5, stall_generations=5, pop_size=500, n_ensemble=10, leverage=3, fitness_config=None):

    # set fitness configuration if provided
    if fitness_config:
        go.set_fitness_config(fitness_config)

    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    total_months = (end_date_dt.year - start_date_dt.year) * 12 + (end_date_dt.month - start_date_dt.month)
    num_periods = (total_months - in_sample_months) // out_sample_months + 1
    
    wfo_results = []
    all_ensemble_strategies = []
    all_performances = []
    cumulative_values = [] 
    
    engine = create_engine()
    df = connect(engine, "test_data")
    
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD OPTIMIZATION WITH ENSEMBLE")
    print(f"In-sample period: {in_sample_months} months, Out-of-sample period: {out_sample_months} months")
    print(f"Ensemble size: {n_ensemble} strategies per period")
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
            pop_size=pop_size, 
            leverage=leverage,
            fitness_config=fitness_config
        )
        
        # select diverse strategies for ensemble
        ensemble_strategies = select_diverse_strategies(pareto_front, n_strategies=n_ensemble)
        
        print(f"\nEnsemble strategies for period {period+1}:")
        for i, strategy in enumerate(ensemble_strategies):
            window, entry, exit_, sell_threshold, *pos_sizing = strategy
            print(f"\nStrategy {i+1}: RSI={window}, Entry={entry}, Exit={exit_}, "
                  f"Sell={sell_threshold}% \nPosition Sizing={[f'{x:.2f}' for x in pos_sizing]}")
            print("Fitness: [" + ", ".join(f"{v:.2f}" for v in strategy.fitness.values) + "]")
        
        # set initial capital per strategy
        if period == 0:
            in_sample_end_dt = pd.to_datetime(in_sample_end_str)
            spx_after_date = df["SPX Close"][in_sample_end_dt:].iloc[:5]
            if not spx_after_date.empty:
                total_capital = spx_after_date.iloc[0] * n_ensemble
                capital_per_strategy = total_capital / n_ensemble
                print(f"   Total starting capital: ${total_capital:.2f}")
                print(f"   Capital per strategy: ${capital_per_strategy:.2f}")
            else:
                total_capital = 100.0 * n_ensemble
                capital_per_strategy = 100.0
                print(f"   Total starting capital: ${total_capital:.2f} (default)")
                print(f"   Capital per strategy: ${capital_per_strategy:.2f}")
        else:
            total_capital = current_portfolio_value
            capital_per_strategy = total_capital / n_ensemble
            print(f"   Total available capital: ${total_capital:.2f}")
            print(f"   Capital per strategy: ${capital_per_strategy:.2f}")
        
        # run ensemble backtest on out-of-sample period
        combined_values, original_portfolios = run_ensemble_backtest(
            ensemble_strategies, 
            in_sample_end_str, 
            out_sample_end_str, 
            capital_per_strategy,
            leverage=leverage
        )
        
        if combined_values is None or len(combined_values) == 0:
            print(f"No valid ensemble results for period {period+1}")
            continue
        
        cumulative_values.append(combined_values)
        current_portfolio_value = combined_values.iloc[-1]
        
        # calculate performance metrics for the ensemble
        period_return = (combined_values.iloc[-1] / combined_values.iloc[0]) - 1
        
        # calculate other metrics using the combined series
        combined_returns = combined_values.pct_change().dropna()
        if len(combined_returns) > 0:
            annualized_return = (1 + combined_returns.mean()) ** 252 - 1
            volatility = combined_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # calculate max drawdown
            rolling_max = combined_values.cummax()
            drawdown = (combined_values - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # calculate Sortino ratio
            negative_returns = combined_returns[combined_returns < 0]
            downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0
        else:
            annualized_return = 0
            max_drawdown = 0
            sharpe_ratio = 0
            sortino_ratio = 0
        
        performance = {
            'period': period + 1,
            'in_sample_start': in_sample_start_str,
            'in_sample_end': in_sample_end_str,
            'out_sample_end': out_sample_end_str,
            'ensemble_strategies': ensemble_strategies,
            'combined_values': combined_values,
            'final_value': current_portfolio_value,
            'period_return': period_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'n_strategies': len(ensemble_strategies)
        }
        
        wfo_results.append(performance)
        all_ensemble_strategies.append(ensemble_strategies)
        all_performances.append(original_portfolios)
        
        print(f"\nEnsemble out-of-sample performance:")
        print(f"   Starting Value: ${combined_values.iloc[0]:.2f}")
        print(f"   Final Value: ${performance['final_value']:.2f}")
        print(f"   Period Return: {performance['period_return']:.2%}")
        print(f"   Annualized Return: {performance['annualized_return']:.2%}")
        print(f"   Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
        print(f"   Sortino Ratio: {performance['sortino_ratio']:.3f}")
        print(f"   Number of strategies: {performance['n_strategies']}")
        
        if period < num_periods - 1:
            current_start = current_start + pd.DateOffset(months=out_sample_months)
    
    return wfo_results, all_ensemble_strategies, all_performances, cumulative_values

def analyze_wfo(wfo_results, all_ensemble_strategies, all_performances, cumulative_values, start_date, end_date):
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
    df = connect(engine, "test_data")
    
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

def main():
    start_date = "1995-01-01"
    end_date = "2009-12-31"
    
    # create custom fitness configuration
    custom_config = FitnessConfig(
        selected_metrics=['sharpe', 'alpha'],  # remove position_stability
        custom_weights={'sharpe': 1.0, 'alpha': 3.0},  # give more weight to alpha
        enable_bottom_percentile_filter=True,
        bottom_percentile=10.0
    )
    
    print("Starting Walk-Forward Optimization with Ensemble...")
    print(f"Full period: {start_date} to {end_date}")
    print(f"Selected metrics: {custom_config.selected_metrics}")
    print(f"Weights: {custom_config.get_weights()}")
    
    wfo_results, all_ensemble_strategies, all_performances, cumulative_values = walk_forward_optimization(
        start_date=start_date,
        end_date=end_date,
        in_sample_months=120,
        out_sample_months=24,
        max_time_minutes=20,
        stall_generations=10,
        pop_size=2500,
        n_ensemble=10,
        leverage=3,
        fitness_config=custom_config
    )
    
    analyze_wfo(wfo_results, all_ensemble_strategies, all_performances, cumulative_values, start_date, end_date)

if __name__ == "__main__":
    main()