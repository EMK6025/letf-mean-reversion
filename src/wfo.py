import go
from backtest import Portfolio, Params
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from engine import create_engine, connect
import vectorbt as vbt
import warnings
import random

warnings.filterwarnings("ignore", category=FutureWarning, module='vectorbt')
vbt.settings.array_wrapper['freq'] = '1D'

# setting random seed
seed = 50
random.seed(seed)
np.random.seed(seed)

def walk_forward_optimization(start_date, end_date, max_time_minutes=10, stall_generations=10, pop_size=500):
    """
    Perform walk-forward optimization to find the best strategy for 20 years after start_date.
    
    Args:
        start_date: Starting date for optimization (format: "YYYY-MM-DD")
        max_time_minutes: Maximum time to run optimization in minutes
        stall_generations: Stop if no improvement for this many generations
        pop_size: Population size for genetic algorithm
    
    Returns:
        best_individual: The best strategy found
        best_fitness: Fitness score of best strategy
        generation_count: Number of generations run
    """
    
    # Initialize timing and tracking variables
    start_time = time.time()
    max_time_seconds = max_time_minutes * 60
    
    best_fitness_history = []
    stall_counter = 0
    generation = 0
    
    # Generate initial population
    print("Generating initial population...")
    population = go.create_initial_population(pop_size=pop_size)
    population = go.run_population(population, start_date=start_date, end_date=end_date)
    
    # Track best individual
    valid_pop = [ind for ind in population if ind.fitness.values[0] > -1000]
    if not valid_pop:
        print("No valid strategies found in initial population!")
        return None, -1000, 0, 0
    
    best_individual = max(valid_pop, key=lambda x: x.fitness.values[0])
    best_fitness = best_individual.fitness.values[0]
    best_fitness_history.append(best_fitness)
    
    print(f"\nGeneration {generation}: Best Fitness = {best_fitness:.2f} ({len(valid_pop)}/{pop_size} valid)")
    
    # Evolution loop
    while True:
        generation += 1
        elapsed_time = time.time() - start_time
        
        # Check time limit
        if elapsed_time > max_time_seconds:
            print(f"Time limit reached ({max_time_minutes} minutes)")
            break
        
        print(f"\nGeneration {generation} (Elapsed: {elapsed_time/60:.1f}min)...")
        
        # Create next generation
        population = go.create_next_generation(population)
        population = go.run_population(population, start_date=start_date, end_date=end_date)
        
        # Find best in current generation
        valid_pop = [ind for ind in population if ind.fitness.values[0] > -1000]
        if valid_pop:
            current_best = max(valid_pop, key=lambda x: x.fitness.values[0])
            current_fitness = current_best.fitness.values[0]
            
            # Check for improvement
            if current_fitness > best_fitness:
                improvement = current_fitness - best_fitness
                stall_counter = 0
                best_individual = current_best
                best_fitness = current_fitness
                print(f"NEW BEST! Fitness = {best_fitness:.2f} (+{improvement:.2f}) ({len(valid_pop)}/{pop_size} valid)")
            else:
                stall_counter += 1
                print(f"Overall Best = {best_fitness:.2f} (Stall: {stall_counter}/{stall_generations}) ({len(valid_pop)}/{pop_size} valid)")
        else:
            stall_counter += 1
            print(f"No valid strategies (Stall: {stall_counter}/{stall_generations})")
        
        best_fitness_history.append(best_fitness)
        
        # Check stall condition
        if stall_counter >= stall_generations:
            print(f"Optimization stalled for {stall_generations} generations")
            break
    
    total_time = time.time() - start_time
    print(f"\nTotal optimization time: {total_time/60:.2f} minutes")
    print(f"Final generation: {generation}")
    
    return best_individual, best_fitness, generation, best_fitness_history

def analyze_best_strategy(best_individual, start_date):
    """
    Analyze and display the performance of the best strategy found.
    """
    if best_individual is None:
        print("No valid strategy to analyze!")
        return
    
    print("\n" + "="*60)
    print("BEST STRATEGY ANALYSIS")
    print("="*60)
    
    # Extract parameters
    window, entry, exit_, sell_threshold, *pos_sizing = best_individual
    
    print(f"Strategy Parameters:")
    print(f"   RSI Window: {window}")
    print(f"   Entry Threshold: {entry}")
    print(f"   Exit Threshold: {exit_}")
    print(f"   Sell Threshold: {sell_threshold}%")
    print(f"   Position Sizing: {[round(x, 3) for x in pos_sizing]}")
    print(f"   Fitness Score: {best_individual.fitness.values[0]:.4f}")
    
    # Create parameter object
    best_params = Params(
        window=window,
        entry=entry,
        exit=exit_,
        sell_threshold=sell_threshold,
        position_sizing=np.array(pos_sizing)
    )
    
    # Calculate test period (20 years from start_date)
    start_dt = pd.to_datetime(start_date)
    end_dt = start_dt + pd.DateOffset(years=20)
    end_date = end_dt.strftime("%Y-%m-%d")
    
    print(f"\nPerformance Analysis ({start_date} to {end_date}):")
    
    # Load data for benchmarks
    engine = create_engine()
    df = connect(engine, "test_data")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    
    # Filter to test period
    spx_data = df["SPX Close"].loc[start_date:end_date]
    letf_data = df["3x LETF"].loc[start_date:end_date]
    
    # Create benchmark portfolios
    spx_portfolio = Portfolio.from_holding(close=spx_data, freq='1D')
    letf_portfolio = Portfolio.from_holding(close=letf_data, freq='1D')
    
    # Run strategy
    import backtest
    strategy_portfolio = backtest.run([best_params], start_date, end_date)
    
    # Calculate metrics
    strategy_returns = strategy_portfolio.returns().iloc[:, 0]
    spx_returns = spx_portfolio.returns()
    letf_returns = letf_portfolio.returns()
    
    strategy_annual_return = strategy_portfolio.annualized_return().iloc[0]
    spx_annual_return = spx_portfolio.annualized_return()
    letf_annual_return = letf_portfolio.annualized_return()
    
    strategy_sharpe = strategy_returns.vbt.returns.sharpe_ratio()
    spx_sharpe = spx_returns.vbt.returns.sharpe_ratio()
    letf_sharpe = letf_returns.vbt.returns.sharpe_ratio()
    
    strategy_sortino = strategy_returns.vbt.returns.sortino_ratio()
    spx_sortino = spx_returns.vbt.returns.sortino_ratio()
    letf_sortino = letf_returns.vbt.returns.sortino_ratio()
    
    strategy_omega = go.omega_ratio(strategy_returns, spx_returns)
    spx_omega = go.omega_ratio(spx_returns, spx_returns)
    letf_omega = go.omega_ratio(letf_returns, spx_returns)
    
    strategy_max_dd = strategy_portfolio.max_drawdown().iloc[0]
    spx_max_dd = spx_portfolio.max_drawdown()
    letf_max_dd = letf_portfolio.max_drawdown()
    
    # Print comparison table
    print(f"{'Metric':<20} {'Strategy':<12} {'SPX':<12} {'3x LETF':<12}")
    print("-" * 60)
    print(f"{'Annual Return':<20} {strategy_annual_return:>11.2%} {spx_annual_return:>11.2%} {letf_annual_return:>11.2%}")
    print(f"{'Sharpe Ratio':<20} {strategy_sharpe:>11.3f} {spx_sharpe:>11.3f} {letf_sharpe:>11.3f}")
    print(f"{'Sortino Ratio':<20} {strategy_sortino:>11.3f} {spx_sortino:>11.3f} {letf_sortino:>11.3f}")
    print(f"{'Omega Ratio':<20} {strategy_omega:>11.2%} {spx_omega:>11.2%} {letf_omega:>11.2%}")
    print(f"{'Max Drawdown':<20} {strategy_max_dd:>11.2%} {spx_max_dd:>11.2%} {letf_max_dd:>11.2%}")
    
    # Create performance chart
    strategy_value = strategy_portfolio.value().iloc[:, 0]
    spx_value = spx_portfolio.value()
    letf_value = letf_portfolio.value()
    
    plt.figure(figsize=(14, 8))
    plt.plot(strategy_value.index, strategy_value.values, label='Optimized Strategy', linewidth=2, color='green')
    plt.plot(spx_value.index, spx_value.values, label='SPX', linewidth=2, color='blue')
    plt.plot(letf_value.index, letf_value.values, label='3x LETF', linewidth=2, color='red')
    
    plt.title(f'Strategy Performance Comparison ({start_date} to {end_date})', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    
    return best_params

if __name__ == "__main__":
    # Test with 1990-01-01 start date
    start_date = "1990-01-01"
    end_date = "2009-12-31"
    
    print("Starting Walk-Forward Optimization...")
    print(f"Testing period: 20 years from {start_date}")
    
    # Run optimization
    best_individual, best_fitness, generations, fitness_history = walk_forward_optimization(
        start_date=start_date,
        end_date=end_date,
        max_time_minutes=10,
        stall_generations=10,
        pop_size=5000
    )
    
    if best_individual is not None:
        best_params = analyze_best_strategy(best_individual, start_date)
