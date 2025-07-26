import pandas as pd
import numpy as np
import time
from vectorbt import Portfolio
import vectorbt as vbt
import warnings
import random
from deap import tools
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import backtest
from backtest import Params
import go
from fitness import FitnessConfig, calc_metrics
from backtest_analysis import analyze_wfo
from wfo_sql import insert_new_run, insert_period_summary, insert_period_strategies
from engine import create_engine, connect_time_series

warnings.filterwarnings("ignore", category=FutureWarning, module='vectorbt')
warnings.filterwarnings("ignore", category=RuntimeWarning)

vbt.settings.array_wrapper['freq'] = '1D'

seed = 24
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

        fitness_matrix = np.vstack([ind.fitness.values for ind in front])
        ref = np.array(ref_point)            # shape (n_metrics,)
        
        minimize_mask = np.array([
            fitness_config.is_minimize_metric(m)
            for m in fitness_config.selected_metrics
        ])         # shape (n_metrics,)
        
        diffs = np.where(minimize_mask,
                        ref - fitness_matrix,    # for minimize
                        fitness_matrix - ref)    # for maximize
        diffs_clipped = np.clip(diffs, a_min=0, a_max=None)
        
        volumes = np.prod(diffs_clipped, axis=1)  # shape (n_inds,)
        valid_mask = fitness_matrix[:, 0] > -1000
        total_volume = volumes[valid_mask].sum()

        return total_volume
    
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
    engine = create_engine()
    
    run_id = insert_new_run(engine, start_date, end_date, in_sample_months, 
                   out_sample_months, pop_size, n_ensemble, 
                   leverage, fitness_config)
    
    # set fitness configuration if provided
    if fitness_config:
        go.set_fitness_config(fitness_config)

    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)
    
    total_months = (end_date_dt.year - start_date_dt.year) * 12 + (end_date_dt.month - start_date_dt.month)
    num_periods = (total_months - in_sample_months) // out_sample_months + 1
    
    wfo_results = []
    all_ensemble_strategies = []
    cumulative_values = [] 
    
    df = connect_time_series(engine, "test_data")
    spxt = df["SPX Close"]
    benchmark = Portfolio.from_holding(close=spxt, freq='1D')
    
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
        out_sample_start = in_sample_end + pd.DateOffset(days=1)
        out_sample_end = in_sample_end + pd.DateOffset(months=out_sample_months)
        
        if out_sample_end > end_date_dt:
            out_sample_end = end_date_dt
        
        in_sample_start_str = current_start.strftime('%Y-%m-%d')
        in_sample_end_str = in_sample_end.strftime('%Y-%m-%d')
        out_sample_end_str = out_sample_end.strftime('%Y-%m-%d')
        
        print(f"\n{'='*80}")
        print(f"PERIOD {period+1}/{num_periods}")
        print(f"In-sample: {in_sample_start_str} to {in_sample_end_str}")
        print(f"Out-of-sample: {out_sample_start} to {out_sample_end_str}")
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
        ensemble_count = len(ensemble_strategies)
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
                total_capital = spx_after_date.iloc[0] * ensemble_count
                capital_per_strategy = total_capital / ensemble_count
                print(f"   Total starting capital: ${total_capital:.2f}")
                print(f"   Capital per strategy: ${capital_per_strategy:.2f}")
            else:
                total_capital = 100.0 * ensemble_count
                capital_per_strategy = 100.0
                print(f"   Total starting capital: ${total_capital:.2f} (default)")
                print(f"   Capital per strategy: ${capital_per_strategy:.2f}")
        else:
            total_capital = current_portfolio_value
            capital_per_strategy = total_capital / ensemble_count
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
        
        combined_metrics = calc_metrics(combined_values, benchmark, ensemble_strategies)
        
        period_return = (combined_values.iloc[-1] / combined_values.iloc[0]) - 1
            
        performance = {
            'period': period + 1,
            'in_sample_start': in_sample_start_str,
            'in_sample_end': in_sample_end_str,
            'out_sample_end': out_sample_end_str,
            'ensemble_strategies': ensemble_strategies,
            'combined_values': combined_values,
            'final_value': current_portfolio_value,
            'period_return': period_return,
            'rel_drawdown': combined_metrics['rel_drawdown'],
            'max_drawdown': combined_metrics['drawdown'],
            'sharpe_ratio': combined_metrics['sharpe'],
            'sortino_ratio': combined_metrics['sortino'],
            'n_strategies': len(ensemble_strategies)
        }
        
        wfo_results.append(performance)
        all_ensemble_strategies.append(ensemble_strategies)
        
        print(f"\nEnsemble out-of-sample performance:")
        print(f"   Starting Value: ${combined_values.iloc[0]:.2f}")
        print(f"   Final Value: ${performance['final_value']:.2f}")
        print(f"   Period Return: {performance['period_return']:.2%}")
        print(f"   Annualized Return: {performance['annualized_return']:.2%}")
        print(f"   Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"   Rel Drawdown: {performance['rel_drawdown']:.2%}")
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
        print(f"   Sortino Ratio: {performance['sortino_ratio']:.3f}")
        print(f"   Number of strategies: {performance['n_strategies']}")
        
        if period < num_periods - 1:
            current_start = current_start + pd.DateOffset(months=out_sample_months)
        
        period_id = insert_period_summary(engine, run_id, period, current_start, 
                          in_sample_end, out_sample_end, 
                          pop_size, performance['n_strategies'])
        
        # pareto_front is a List[Individual]
        insert_period_strategies(engine, run_id, period_id, pareto_front)

    # wfo_results = List[Dict]
    # all_ensemble_strategies = List[List[individual]]
    # cumulative_values = List[pd.Series]
    return wfo_results, all_ensemble_strategies, cumulative_values

def main():
    start_date = "1991-01-01"
    end_date = "2024-12-31"
    
    # create custom fitness configuration
    custom_config = FitnessConfig(
        selected_metrics=['sortino', 'sharpe', 'rel_drawdown', 'alpha', 'position_stability', 'var'],  # remove position_stability
        enable_bottom_percentile_filter=True,
        bottom_percentile=10.0
    )
    
    print("Starting Walk-Forward Optimization with Ensemble...")
    print(f"Full period: {start_date} to {end_date}")
    print(f"Selected metrics: {custom_config.selected_metrics}")
    print(f"Weights: {custom_config.get_weights()}")
    
    wfo_results, all_ensemble_strategies, cumulative_values = walk_forward_optimization(
        start_date=start_date,
        end_date=end_date,
        in_sample_months=120,
        out_sample_months=12,
        max_time_minutes=20,
        stall_generations=10,
        pop_size=1000,
        n_ensemble=10,
        leverage=4,
        fitness_config=custom_config
    )
    
    analyze_wfo(wfo_results, all_ensemble_strategies, cumulative_values, start_date, end_date)

if __name__ == "__main__":
    main()