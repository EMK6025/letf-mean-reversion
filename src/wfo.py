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

def walk_forward_optimization(start_date, end_date, max_time_minutes=10, stall_generations=10, pop_size=500):
    from fitness import fitness

    start_time = time.time()
    max_time_seconds = max_time_minutes * 60
    
    hypervolume_history = []
    stall_counter = 0
    generation = 0
    best_hypervolume = 0
    
    print("Generating initial population...")
    population = go.create_initial_population(pop_size=pop_size)
    
    # Get pareto front from initial population
    print("Calculating initial Pareto front...")
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    
    # Calculate hypervolume to proxy evolution progression
    # Reference point for hypervolume (worst case values)
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

    # Create performance comparison plot
    plt.figure(figsize=(14, 8))
    
    # Plot benchmarks first
    plt.plot(etf1_value.index, etf1_value.values, 
             label='1x SPX', color='blue', linewidth=2, alpha=0.8)
    plt.plot(etf3_value.index, etf3_value.values, 
             label='3x LETF', color='green', linewidth=2, alpha=0.8)
    
    # Plot all Pareto front strategies in red with varying transparency
    for i, pf in enumerate(pfs):
        alpha = 0.3 + 0.4 * (i / pfs.value().shape[1])  # Varying transparency
        plt.plot(pf.value().index, pf.value().values, 
                color='red', alpha=alpha, linewidth=1)
    
    # Add a single legend entry for Pareto strategies
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

    # Summary statistics for Pareto front
    final_values = [pf.value().iloc[-1] for pf in pfs]
    annual_returns = [pf.annualized_return() for pf in pfs]
    max_drawdowns = [pf.max_drawdown() for pf in pfs]
    
    print(f"\nPareto Front Performance Summary:")
    print(f"Final Values - Min: ${min(final_values):,.0f}, Max: ${max(final_values):,.0f}, Median: ${np.median(final_values):,.0f}")
    print(f"Annual Returns - Min: {min(annual_returns):.1%}, Max: {max(annual_returns):.1%}, Median: {np.median(annual_returns):.1%}")
    print(f"Max Drawdowns - Min: {min(max_drawdowns):.1%}, Max: {max(max_drawdowns):.1%}, Median: {np.median(max_drawdowns):.1%}")
    
    return balanced_strategy

if __name__ == "__main__":
    start_date = "1990-01-01"
    end_date = "2009-12-31"
    
    print("Starting Multi-Objective Walk-Forward Optimization...")
    print(f"Testing period: {start_date} to {end_date}")
    
    pareto_front, generations, hypervolume_history = walk_forward_optimization(
        start_date=start_date,
        end_date=end_date,
        max_time_minutes=10,
        stall_generations=10,
        pop_size=1000
    )
    
    if pareto_front:
        plt.figure(figsize=(10, 6))
        plt.plot(hypervolume_history)
        plt.title('Hypervolume Progress')
        plt.xlabel('Generation')
        plt.ylabel('Hypervolume')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        balanced_strategy = analyze_pareto_front(pareto_front, start_date)