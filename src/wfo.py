import pandas as pd
import numpy as np
import time
from vectorbt import settings
import warnings

from backtest import run, Params
from fitness import calc_metrics
from engine import create_engine, connect_time_series

warnings.filterwarnings('ignore', category=FutureWarning, module='vectorbt')
warnings.filterwarnings('ignore', category=RuntimeWarning)

settings.array_wrapper['freq'] = '1D'

import random

MIN_GEN = 75

engine = create_engine()
df = connect_time_series(engine, 'test_data')
spxt = df['SPX Close']

import sys

_first = True

def print_replace(line: str):
    global _first
    if not _first:
        sys.stdout.write('\x1b[1A')  # move cursor up
        sys.stdout.write('\x1b[2K')  # clear line
    sys.stdout.write(line + '\n')
    sys.stdout.flush()
    _first = False

def window_backtest(ref_points, start_date, end_date, max_time_minutes=10, stall_generations=10, max_generations=100, pop_size=500, leverage=3, fitness_config=None):

    from go import create_initial_population, create_next_generation
    from deap import tools
    
    start_time = time.time()
    max_time_seconds = max_time_minutes * 60
    
    hypervolume_history = []
    stall_counter = 0
    generation = 0
    best_hypervolume = 0
    
    # print('Generating initial population...')
    population = create_initial_population(pop_size=pop_size, start_date=start_date, end_date=end_date, leverage=leverage)

    # print('Calculating initial Pareto front...')
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
         
        valid_mask = (fitness_matrix[:, 0] > -1000) & np.isfinite(diffs_clipped).all(axis=1)

        total_volume = volumes[valid_mask].sum()
        return total_volume
    
    # print('Calculating initial hypervolume...')
    current_hypervolume = calculate_hypervolume(pareto_front, ref_points)
    best_hypervolume = current_hypervolume
    hypervolume_history.append(current_hypervolume)
    
    # print(f'\nGeneration {generation}: Pareto front size = {len(pareto_front)}, Hypervolume = {current_hypervolume:.4f}')
    global _first
    _first = True
    while generation < max_generations:
        if best_hypervolume > 0:
            generation += 1
        elapsed_time = time.time() - start_time
        
        if elapsed_time > max_time_seconds:
            print(f'Time limit reached ({max_time_minutes} minutes)')
            break
        
        # print(f'\nGeneration {generation} (Elapsed: {elapsed_time/60:.1f}min)...')
        
        population = create_next_generation(population, start_date=start_date, end_date=end_date, leverage=leverage)
        
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        current_hypervolume = calculate_hypervolume(pareto_front, ref_points)
        if current_hypervolume > best_hypervolume:
            improvement = current_hypervolume - best_hypervolume
            stall_counter = 0
            best_hypervolume = current_hypervolume
            print_replace(f'Generation {generation}: NEW BEST! Hypervolume = {current_hypervolume:.4f} (+{improvement:.4f}), Pareto size = {len(pareto_front)}')
        elif generation == 0:
            print_replace(f'Generation {generation}: Hypervolume = {current_hypervolume:.4f}, Pareto size = {len(pareto_front)}')
            pass
        else:
            if generation < MIN_GEN:
                print_replace(f'Generation {generation}: Hypervolume = {current_hypervolume:.4f} (Stall: {stall_counter}/{stall_generations}), Pareto size = {len(pareto_front)}')
            else:
                stall_counter += 1
                print_replace(f'Generation {generation}: Hypervolume = {current_hypervolume:.4f} (Stall: {stall_counter}/{stall_generations}), Pareto size = {len(pareto_front)}')
        
        hypervolume_history.append(current_hypervolume)
        
        if stall_counter >= stall_generations and generation > MIN_GEN:
            break
    
    # total_time = time.time() - start_time
    # print(f'\nTotal optimization time: {total_time/60:.2f} minutes')
    # print(f'Final generation: {generation}')
    # print(f'Final Pareto front size: {len(pareto_front)}')
    
    return pareto_front, generation, best_hypervolume

def select_diverse_strategies(pareto_front, n_strategies=10, seed=1):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans  
    if len(pareto_front) <= n_strategies:
        # print(f'Pareto front has only {len(pareto_front)} strategies, using all of them')
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

    '''
    select one strategy from n random clusters
    '''
    # use K-means clustering to find diverse strategies
    kmeans = KMeans(n_clusters=n_strategies, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(strategy_params_scaled)
    
    diverse_strategies = []
        
    for cluster_id in np.unique(labels):
        cluster_idx = np.where(labels == cluster_id)[0]
        
        if cluster_idx.size == 0:
            continue
        cluster_strategies = strategy_params_scaled[cluster_idx]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        j = np.argmin(np.linalg.norm(cluster_strategies - cluster_center, axis=1))
        diverse_strategies.append(pareto_front[cluster_idx[j]])
    
    # print(f'Selected {len(diverse_strategies)} diverse strategies from {len(pareto_front)} Pareto front strategies')
    
    return diverse_strategies

def run_ensemble_backtest(cleaned_pareto_front, start_date, target_backtest_duration_months, current_portfolio_value=10000, n_strategies=10, leverage=3, fitness_config=None, seed=1):
    '''
    Run backtest for ensemble of strategies with equal capital allocation.
    output: List[List] ensemble_strategies (with updated fitness) and the ending portfolio value
    '''
    from vectorbt import Portfolio
    from math import ceil, isfinite
    
    tentative_end_date = start_date + pd.DateOffset(months=ceil(target_backtest_duration_months * 1.5))
    target_end_date = start_date + pd.DateOffset(months=target_backtest_duration_months)
    start_date_str = start_date.strftime('%Y-%m-%d')
    tentative_end_date_str = tentative_end_date.strftime('%Y-%m-%d')
    
    if len(cleaned_pareto_front) > n_strategies:
        ensemble_strategies = select_diverse_strategies(cleaned_pareto_front, n_strategies=n_strategies, seed=seed)
    else:
        ensemble_strategies = cleaned_pareto_front
    
    ensemble_count = len(ensemble_strategies)
    
    # set initial capital per strategy
    # print(f'   Total available capital: ${current_portfolio_value:.2f}')
    # print(f'   Capital per strategy: ${capital_per_strategy:.2f}')
        
    # run ensemble backtest on out-of-sample period
    # print(f'Running ensemble backtest with {len(ensemble_strategies)} strategies...')

    ensemble_params = []
    for individual in ensemble_strategies:
        window, entry, exit, sell_threshold, *pos_sizing = individual
        ensemble_params.append(Params(
            window=window,
            entry=entry,
            exit=exit,
            sell_threshold=sell_threshold,
            position_sizing=np.array(pos_sizing)
        ))

    capital_per_strategy = current_portfolio_value / ensemble_count

    pfs = run(ensemble_params, start_date_str, tentative_end_date_str, target_end_date, capital_per_strategy, leverage=leverage)

    records = pfs.trades.records_readable.copy()
    records = records[records['Status'] == 'Closed']['Exit Timestamp']
    end_date = records.max()
    if pd.isna(end_date):
        end_date = target_end_date
    end_date_str = end_date.strftime('%Y-%m-%d')

    benchmark = Portfolio.from_holding(close=spxt[start_date_str:end_date_str], freq='1D')
    rf = df['RF Rate'][start_date_str:end_date_str]
    
    # recast to relevant send_date slice    
    sliced_pfs = Portfolio.from_holding(
        close = pfs.value().loc[:end_date_str],
        size  = 1,
        freq  = pfs.wrapper.freq
    )
    
    # find out-of-sample fitness for ensemble strategies
    all_metrics = calc_metrics(sliced_pfs, benchmark, rf, ensemble_params)
    
    # rewrite fitness with out-of-sample values    
    for i, individual in enumerate(ensemble_strategies):
        new_fitness_values = []
        for metric in all_metrics:
            if metric in fitness_config.selected_metrics:
                metric_values = all_metrics[metric]
                value = metric_values.iloc[i]
                if not isfinite(value):
                    value = 0.0
                new_fitness_values.append(float(value))
        individual.fitness.values = tuple(new_fitness_values)

    combined_performance = pfs.value().sum(axis=1).loc[:end_date_str]
    
    combined_pf = Portfolio.from_holding(
        close = combined_performance,
        size  = 1,
        freq  = pfs.wrapper.freq
    )
    
    i_start = spxt.index.searchsorted(start_date_str, side='left') 
    i_end = spxt.index.searchsorted(end_date_str, side='right') - 1 
    benchmark_start = spxt.iloc[i_start] 
    benchmark_end = spxt.iloc[i_end]    
    period_return   = (combined_performance.iloc[-1] / combined_performance.iloc[0]) - 1
    combined_metrics = calc_metrics(combined_pf, benchmark, rf, ensemble_params)
    print(f'supposed starting value (current_portfolio_value) is ${current_portfolio_value}')
    print(f'\nActual out-of-sample range: {start_date_str} to {end_date_str}')
    print(f'\nEnsemble out-of-sample performance:')
    print(f'   Starting Value: ${combined_performance.iloc[0]:.2f}')
    print(f'   Final Value: ${combined_performance.iloc[-1]:.2f}')
    print(f'   Starting Benchmark Value: ${benchmark_start:.2f}')
    print(f'   Final Benchmark Value: ${benchmark_end:.2f}')
    print(f'   Period Return: {period_return:.2%}')
    print(f'   Benchmark Return: {benchmark.total_return():.2%}')
    print(f'   Max Drawdown: {combined_metrics['drawdown']:.2%}')
    print(f'   Rel Drawdown: {combined_metrics['rel_drawdown']:.2%}')
    print(f'   Sharpe Ratio: {combined_metrics['sharpe']:.3f}')
    print(f'   Sortino Ratio: {combined_metrics['sortino']:.3f}')
    print(f'   Beta: {combined_metrics['beta']}')
    print(f'   Number of strategies: {len(ensemble_strategies)}')
    
    
    return ensemble_strategies, combined_performance, end_date

def walk_forward_optimization(start_date, end_date, in_sample_months=60, out_sample_months=12, 
                             max_time_minutes=5, stall_generations=5, max_generations = 100, 
                             pop_size=500, n_ensemble=10, leverage=3, fitness_config=None, rand_seed=1):    
    from go import set_fitness_config
    from wfo_sql import insert_new_run, insert_period_summary, insert_period_strategies
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    run_id = insert_new_run(engine, start_date, end_date, in_sample_months, 
            out_sample_months, pop_size, n_ensemble, 
            leverage, fitness_config)
    
    # set fitness configuration if provided
    if fitness_config:
        set_fitness_config(fitness_config)

    # dynamic reference point based on selected metrics
    ref_points = []
    for metric in fitness_config.selected_metrics:
        params = fitness_config.available_metrics[metric]
        ref_points.append(params[2])
        
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    print(f'\n{'='*80}')
    print(f'WALK-FORWARD OPTIMIZATION WITH ENSEMBLE')
    print(f'In-sample period: {in_sample_months} months, Out-of-sample period: {out_sample_months} months')
    print(f'Ensemble size: {n_ensemble} strategies per period')
    print(f'seed: {rand_seed}')
    print(f'{'='*80}\n')
    
    current_start = start_date
    first_in_sample_end = current_start + pd.DateOffset(months=in_sample_months)
    spx_after_date = df['SPX Close'][first_in_sample_end:].iloc[0]
    current_portfolio_value = spx_after_date
    
    period = 1
    
    while True:
        in_sample_end = current_start + pd.DateOffset(months=in_sample_months)
        out_sample_start = in_sample_end + pd.DateOffset(days=1)
        out_sample_end = out_sample_start + pd.DateOffset(months=out_sample_months)
        
        if out_sample_end > end_date:
            out_sample_end = end_date

        if out_sample_start >= end_date: 
            break
        in_sample_start_str = current_start.strftime('%Y-%m-%d')
        in_sample_end_str = in_sample_end.strftime('%Y-%m-%d')
        out_sample_start_str = out_sample_start.strftime('%Y-%m-%d')
        out_sample_end_str = out_sample_end.strftime('%Y-%m-%d')
        
        print(f'\n{'='*80}')
        print(f'PERIOD {period}')
        print(f'In-sample: {in_sample_start_str} to {in_sample_end_str}')
        print(f'Target Out-of-sample: {out_sample_start_str} to {out_sample_end_str}')
        print(f'{'='*80}\n')
        
        pareto_front, generation_count, hypervolume = window_backtest(
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
        
        ensemble_strategies, combined_performance, actual_end_date = run_ensemble_backtest(
            pareto_front, 
            out_sample_start, 
            out_sample_months, 
            current_portfolio_value,
            n_strategies=n_ensemble,
            leverage=leverage,
            fitness_config=fitness_config, 
            seed=rand_seed
        )
            
        period_id = insert_period_summary(engine, run_id, period, current_start, 
                          in_sample_end, actual_end_date, 
                          pop_size, len(ensemble_strategies), generation_count=generation_count, final_hypervolume = hypervolume
                          )
        
        # pareto_front is a List[Individual]
        insert_period_strategies(engine, run_id, period_id, ensemble_strategies)
        
        current_portfolio_value = combined_performance.iloc[-1]
        current_start = actual_end_date - pd.DateOffset(months=in_sample_months)
        period += 1
