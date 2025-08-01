import pandas as pd
import numpy as np
import time
import vectorbt as vbt
import warnings
import random
from deap import tools
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import backtest
from backtest import Params
import go
from fitness import calc_metrics
from wfo_sql import insert_new_run, insert_period_summary, insert_period_strategies
from engine import create_engine, connect_time_series

warnings.filterwarnings("ignore", category=FutureWarning, module='vectorbt')
warnings.filterwarnings("ignore", category=RuntimeWarning)

vbt.settings.array_wrapper['freq'] = '1D'

seed = 69
random.seed(seed)
np.random.seed(seed)

engine = create_engine()
df = connect_time_series(engine, "test_data")
spxt = df["SPX Close"]

def window_backtest(ref_points, start_date, end_date, max_time_minutes=10, stall_generations=10, max_generations=100, pop_size=500, leverage=3, fitness_config=None):
    
    start_time = time.time()
    max_time_seconds = max_time_minutes * 60
    
    hypervolume_history = []
    stall_counter = 0
    generation = 1
    best_hypervolume = 0
    
    print("Generating initial population...")
    population = go.create_initial_population(pop_size=pop_size, start_date=start_date, end_date=end_date, leverage=leverage)
    
    print("Calculating initial Pareto front...")
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
                
    def calculate_hypervolume(front, ref_points):
        # front is a List[Individual]
        if not front:
            return 0
        fitness_matrix = np.vstack([ind.fitness.values for ind in front])
        ref = np.array(ref_points)            # shape (n_metrics,)
        
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
    current_hypervolume = calculate_hypervolume(pareto_front, ref_points)
    best_hypervolume = current_hypervolume
    hypervolume_history.append(current_hypervolume)
    
    print(f"\nGeneration {generation}: Pareto front size = {len(pareto_front)}, Hypervolume = {current_hypervolume:.4f}")
    
    while generation < max_generations:
        if best_hypervolume != 0:
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
            print(f"Hypervolume = {current_hypervolume:.4f} (Stall: {stall_counter}/{stall_generations}), Pareto size = {len(   )}")
        
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
    
    # when clustering through input space
    scaler = StandardScaler()
    strategy_params_scaled = scaler.fit_transform(strategy_params)
    
    # when clustering through fitness space
    # fitness_matrix = np.vstack([ind.fitness.values for ind in pareto_front])
    # strategy_params_scaled = StandardScaler().fit_transform(fitness_matrix)

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

def run_ensemble_backtest(cleaned_pareto_front, period, start_date, end_date, current_portfolio_value=10000, n_strategies=10, leverage=3):
    """
    Run backtest for ensemble of strategies with equal capital allocation.
    output: Portfolio combined_pf, Series combined_performance, and List[Params] params (processed version of strategies)
    """
    ensemble_strategies = select_diverse_strategies(cleaned_pareto_front, n_strategies=n_strategies)
    ensemble_count = len(ensemble_strategies)
    print(f"\nEnsemble strategies for period {period+1}:")
    for i, strategy in enumerate(ensemble_strategies):
        window, entry, exit_, sell_threshold, *pos_sizing = strategy
        print(f"\nStrategy {i+1}: RSI={window}, Entry={entry}, Exit={exit_}, "
                f"Sell={sell_threshold}% \nPosition Sizing={[f'{x:.2f}' for x in pos_sizing]}")
        print("Fitness: [" + ", ".join(f"{v:.2f}" for v in strategy.fitness.values) + "]")
    
    # set initial capital per strategy
    capital_per_strategy = current_portfolio_value / ensemble_count
    print(f"   Total available capital: ${current_portfolio_value:.2f}")
    print(f"   Capital per strategy: ${capital_per_strategy:.2f}")
        
    # run ensemble backtest on out-of-sample period
    print(f"Running ensemble backtest with {len(ensemble_strategies)} strategies...")
    print(f"Capital per strategy: ${capital_per_strategy:,}")

    ensemble_params = []
    for individual in ensemble_strategies:
        window, entry, exit_, sell_threshold, *pos_sizing = individual
        ensemble_params.append(Params(
            window=window,
            entry=entry,
            exit=exit_,
            sell_threshold=sell_threshold,
            position_sizing=np.array(pos_sizing)
        ))

    pfs = backtest.run(ensemble_params, start_date, end_date, capital_per_strategy, leverage=leverage)
    
    combined_performance = pfs.value().sum(axis=1)
    
    combined_pf = vbt.Portfolio.from_holding(
        close=combined_performance,
        size = 1,
        freq=pfs.wrapper.freq
    )
    portfolio_value = combined_performance.iloc[-1]
    benchmark = vbt.Portfolio.from_holding(close=spxt[start_date:end_date], freq='1D')
    combined_metrics = calc_metrics(combined_pf, benchmark, ensemble_params)
    period_return = (combined_performance.iloc[-1] / combined_performance.iloc[0]) - 1
    
    print(f"\nEnsemble out-of-sample performance:")
    print(f"   Starting Value: ${current_portfolio_value:.2f}")
    print(f"   Final Value: ${portfolio_value:.2f}")
    print(f"   Period Return: {period_return:.2%}")
    print(f"   Benchmark Return: {benchmark.total_return():.2%}")
    print(f"   Max Drawdown: {combined_metrics['drawdown']:.2%}")
    print(f"   Rel Drawdown: {combined_metrics['rel_drawdown']:.2%}")
    print(f"   Sharpe Ratio: {combined_metrics['sharpe']:.3f}")
    print(f"   Sortino Ratio: {combined_metrics['sortino']:.3f}")
    print(f"   Number of strategies: {len(ensemble_strategies)}")
        
    return ensemble_strategies, portfolio_value

def walk_forward_optimization(start_date, end_date, in_sample_months=60, out_sample_months=12, 
                             max_time_minutes=5, stall_generations=5, max_generations = 100, pop_size=500, n_ensemble=10, leverage=3, fitness_config=None):    
    run_id = insert_new_run(engine, start_date, end_date, in_sample_months, 
                   out_sample_months, pop_size, n_ensemble, 
                   leverage, fitness_config)
    
    # set fitness configuration if provided
    if fitness_config:
        go.set_fitness_config(fitness_config)

    # dynamic reference point based on selected metrics
    ref_points = []
    for metric in fitness_config.selected_metrics:
        params = fitness_config.available_metrics[metric]
        ref_points.append(params[2])
        
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    num_periods = (total_months - in_sample_months) // out_sample_months + 1
    
    print(f"\n{'='*80}")
    print(f"WALK-FORWARD OPTIMIZATION WITH ENSEMBLE")
    print(f"In-sample period: {in_sample_months} months, Out-of-sample period: {out_sample_months} months")
    print(f"Ensemble size: {n_ensemble} strategies per period")
    print(f"Total periods: {num_periods}")
    print(f"{'='*80}\n")
    
    current_start = start_date
    first_in_sample_end = current_start + pd.DateOffset(months=in_sample_months)
    spx_after_date = df["SPX Close"][first_in_sample_end:].iloc[0]
    current_portfolio_value = spx_after_date
    
    for period in range(num_periods):
        in_sample_end = current_start + pd.DateOffset(months=in_sample_months)
        out_sample_start = in_sample_end + pd.DateOffset(days=1)
        out_sample_end = in_sample_end + pd.DateOffset(months=out_sample_months)
        
        if out_sample_end > end_date:
            out_sample_end = end_date
        
        in_sample_start_str = current_start.strftime('%Y-%m-%d')
        in_sample_end_str = in_sample_end.strftime('%Y-%m-%d')
        out_sample_start_str = out_sample_start.strftime('%Y-%m-%d')
        out_sample_end_str = out_sample_end.strftime('%Y-%m-%d')
        
        print(f"\n{'='*80}")
        print(f"PERIOD {period+1}/{num_periods}")
        print(f"In-sample: {in_sample_start_str} to {in_sample_end_str}")
        print(f"Out-of-sample: {out_sample_start_str} to {out_sample_end_str}")
        print(f"{'='*80}\n")
        
        pareto_front = window_backtest(
            ref_points,
            start_date=in_sample_start_str,
            end_date=in_sample_end_str,
            max_time_minutes=max_time_minutes,
            stall_generations=stall_generations,
            max_generations=max_generations,
            pop_size=pop_size, 
            leverage=leverage,
            fitness_config=fitness_config
        )
        
        # select diverse strategies for ensemble
        if 'alpha' in fitness_config.selected_metrics:
            alpha_index = fitness_config.selected_metrics.index('alpha')
            cleaned_pareto_front = [ind for ind in pareto_front if ind.fitness.values[alpha_index] > 0]
        else:
            cleaned_pareto_front = pareto_front 
            
        if not cleaned_pareto_front:
            cleaned_pareto_front = pareto_front
        
        ensemble_strategies, portfolio_value = run_ensemble_backtest(
            cleaned_pareto_front, 
            out_sample_start_str, 
            out_sample_end_str, 
            current_portfolio_value,
            leverage=leverage
        )
        
        current_portfolio_value = portfolio_value
        
        if period < num_periods - 1:
            current_start = current_start - pd.DateOffset(months=out_sample_end)
        
        period_id = insert_period_summary(engine, run_id, period, current_start, 
                          in_sample_end, out_sample_end, 
                          pop_size, len(ensemble_strategies))
        
        # pareto_front is a List[Individual]
        insert_period_strategies(engine, run_id, period_id, ensemble_strategies)
