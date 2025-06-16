import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from engine import create_engine, connect
import backtest
from backtest import Params
from vectorbt import Portfolio
import vectorbt as vbt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='vectorbt')
vbt.settings.array_wrapper['freq'] = '1D'

# Fitness thresholds and weights
SORTINO_THRESHOLD = .5
SHARPE_THRESHOLD = .5
OMEGA_THRESHOLD = .85
DRAWDOWN_THRESHOLD = 2
ALPHA_THRESHOLD = .04

SORTINO_WEIGHT = 30
SHARPE_WEIGHT = 30
OMEGA_WEIGHT = 30
ALPHA_WEIGHT = 40

ALPHA_SCALING = 75
DRAWDOWN_SCALING = 2

def omega_ratio(returns, benchmark):
    """Calculate omega ratio comparing returns to benchmark."""
    excess_returns = returns.sub(benchmark, axis=0)
    gains = excess_returns.clip(lower=0)
    losses = excess_returns.clip(upper=0)
    mean_gains = gains.mean(axis=0)
    mean_losses = losses.abs().mean(axis=0)
    if isinstance(mean_losses, pd.Series):
        mean_losses = mean_losses.replace(0, pd.NA)
    else:
        mean_losses = float(mean_losses) if mean_losses != 0 else float("nan")
    return mean_gains / mean_losses

def fitness(sortino, sharpe, omega, relative_drawdowns, alpha):
    """
    Calculate fitness score based on multiple performance metrics.
    
    Args:
        sortino: Sortino ratio values
        sharpe: Sharpe ratio values  
        omega: Omega ratio values
        relative_drawdowns: Relative drawdown values
        alpha: Alpha values (excess return over benchmark)
    
    Returns:
        List of tuples containing fitness scores
    """
    fail_mask = (sortino < SORTINO_THRESHOLD) | (sharpe < SHARPE_THRESHOLD) | (omega < OMEGA_THRESHOLD) | (relative_drawdowns < DRAWDOWN_THRESHOLD)
    
    norm_sortino = (np.minimum(sortino, 3) - SORTINO_THRESHOLD) * SORTINO_WEIGHT / (3 - SORTINO_THRESHOLD)
    norm_sharpe = (np.minimum(sharpe, 3) - SHARPE_THRESHOLD) * SHARPE_WEIGHT / (3 - SHARPE_THRESHOLD)
    norm_omega = (np.minimum(omega, 3) - OMEGA_THRESHOLD) * OMEGA_WEIGHT / (3 - OMEGA_THRESHOLD)
    norm_alpha = 1 / (1 + np.exp(-ALPHA_SCALING * (alpha - ALPHA_THRESHOLD))) * ALPHA_WEIGHT
    norm_drawdown = 1 / (1 + np.exp(-DRAWDOWN_SCALING * (relative_drawdowns - DRAWDOWN_THRESHOLD)))

    fitness_val = (norm_sharpe + norm_omega + norm_alpha) * (norm_drawdown)
    
    fitness_val[fail_mask] = -1000
    return list(zip(fitness_val))

def test_fitness_function(pop_size=10000, seed=42, start_date="1989-12-31", end_date="2020-12-31"):
    """
    Test the fitness function with a large population and analyze top performers.
    
    Args:
        pop_size: Size of test population
        seed: Random seed for reproducibility
        start_date: Start date for backtesting
        end_date: End date for backtesting
    """
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Testing fitness function with {pop_size} strategies...")
    print(f"Test period: {start_date} to {end_date}")
    print("="*80)
    
    # Load data for benchmarks
    engine = create_engine()
    df = connect(engine, "test_data")[["Date","SPX Close","3x LETF"]]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    
    # Generate random population using same logic as go.py
    from go import create_individual, benchmark, benchmark_returns
    
    # Create population
    population = [create_individual() for _ in range(pop_size)]

    
    # Convert to Params objects
    params_list  = [
        Params(window, entry, exit_, sell_threshold, position_sizing=np.array(pos_sizing))
        for window, entry, exit_, sell_threshold, *pos_sizing in population
    ]


    print("Running backtests...")
    # Run backtests
    pfs = backtest.run(params_list, start_date, end_date)
    
    print("Calculating metrics...")
    # Calculate metrics
    returns = pfs.returns()
    sortino = returns.vbt.returns.sortino_ratio()
    sharpe = returns.vbt.returns.sharpe_ratio()
    omega = omega_ratio(returns, benchmark_returns)
    drawdown = benchmark.max_drawdown() / pfs.max_drawdown()
    alpha = pfs.annualized_return() - benchmark.annualized_return()
    annual_return = pfs.annualized_return()

    
    # Calculate fitness scores
    fitness_vals = fitness(sortino, sharpe, omega, drawdown, alpha)
    fitness_scores = [f[0] for f in fitness_vals]

    # Create results dataframe
    results_data = []
    for i, (individual, fitness_score) in enumerate(zip(population, fitness_scores)):
        window, entry, exit_, sell_threshold, *pos_sizing = individual
        
        results_data.append({
            'strategy_id': i,
            'fitness': fitness_score,
            'window': window,
            'entry': entry,
            'exit': exit_,
            'sell_threshold': sell_threshold,
            'position_sizing': pos_sizing,
            'sortino': sortino.iloc[i],
            'sharpe': sharpe.iloc[i],
            'omega': omega.iloc[i],
            'relative_drawdown': drawdown.iloc[i],
            'alpha': alpha.iloc[i],
            'annual_return': annual_return.iloc[i]
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Get top 10 strategies
    top_10 = results_df.nlargest(10, 'fitness')
    
    # Print results
    print("\nTOP 10 STRATEGIES:")
    print("="*120)
    print(f"{'Rank':<4} {'Fitness':<8} {'Window':<6} {'Entry':<5} {'Exit':<4} {'Sell%':<5} {'Sortino':<7} {'Sharpe':<6} {'Omega':<6} {'RelDD':<5} {'Alpha':<6} {'AnnRet':<7}")
    print("-"*120)
    
    output_lines = []
    output_lines.append("TOP 10 STRATEGIES:")
    output_lines.append("="*120)
    output_lines.append(f"{'Rank':<4} {'Fitness':<8} {'Window':<6} {'Entry':<5} {'Exit':<4} {'Sell%':<5} {'Sortino':<7} {'Sharpe':<6} {'Omega':<6} {'RelDD':<5} {'Alpha':<6} {'AnnRet':<7}")
    output_lines.append("-"*120)
    
    for rank, (_, row) in enumerate(top_10.iterrows(), 1):
        line = f"{rank:<4} {row['fitness']:<8.2f} {row['window']:<6} {row['entry']:<5} {row['exit']:<4} {row['sell_threshold']:<5} {row['sortino']:<7.3f} {row['sharpe']:<6.3f} {row['omega']:<6.3f} {row['relative_drawdown']:<5.2f} {row['alpha']:<6.3f} {row['annual_return']:<7.2%}"
        print(line)
        output_lines.append(line)
        
        pos_sizing_str = f"Position Sizing: {[round(x, 3) for x in row['position_sizing']]}"
        print(f"     {pos_sizing_str}")
        output_lines.append(f"     {pos_sizing_str}")
        print()
        output_lines.append("")
    
    # Save to file
    with open('fitness_test_results.txt', 'w') as f:
        f.write('\n'.join(output_lines))
        f.write(f"\n\nTest Parameters:\n")
        f.write(f"Population Size: {pop_size}\n")
        f.write(f"Random Seed: {seed}\n")
        f.write(f"Test Period: {start_date} to {end_date}\n")
        f.write(f"Valid Strategies: {len(results_df[results_df['fitness'] > -1000])}/{pop_size}\n")
    
    print(f"Results saved to fitness_test_results.txt")
    
    # Create performance comparison chart
    print("\nGenerating performance comparison chart...")
    
    # Get top 10 strategy parameters
    top_10_params = []
    for _, row in top_10.iterrows():
        top_10_params.append(Params(
            window=row['window'],
            entry=row['entry'],
            exit=row['exit'],
            sell_threshold=row['sell_threshold'],
            position_sizing=np.array(row['position_sizing'])
        ))
    
    # Run backtests for top 10
    top_10_pfs = backtest.run(top_10_params, start_date, end_date)
    
    # Create benchmarks
    spx_data = df["SPX Close"].loc[start_date:end_date]
    letf_data = df["3x LETF"].loc[start_date:end_date]
    spx_portfolio = Portfolio.from_holding(close=spx_data, freq='1D')
    letf_portfolio = Portfolio.from_holding(close=letf_data, freq='1D')
    
    # Plot performance
    plt.figure(figsize=(16, 10))
    
    # Plot benchmarks
    spx_value = spx_portfolio.value()
    letf_value = letf_portfolio.value()
    plt.plot(spx_value.index, spx_value.values, label='SPX', linewidth=3, color='blue', alpha=0.8)
    plt.plot(letf_value.index, letf_value.values, label='3x LETF', linewidth=3, color='red', alpha=0.8)
    
    # Plot top 10 strategies
    colors = plt.cm.Set3(np.linspace(0, 1, 10))
    for i in range(min(10, len(top_10_params))):
        strategy_value = top_10_pfs.value().iloc[:, i]
        fitness_score = top_10.iloc[i]['fitness']
        plt.plot(strategy_value.index, strategy_value.values, 
                label=f'Strategy {i+1} (Fitness: {fitness_score:.1f})', 
                linewidth=2, color=colors[i], alpha=0.7)
    
    plt.title(f'Top 10 Strategies vs Benchmarks ({start_date} to {end_date})', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('fitness_test_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Performance chart saved as fitness_test_performance.png")
    
    # Summary statistics
    valid_strategies = results_df[results_df['fitness'] > -1000]
    print(f"\nSUMMARY STATISTICS:")
    print(f"Valid Strategies: {len(valid_strategies)}/{pop_size} ({len(valid_strategies)/pop_size:.1%})")
    print(f"Best Fitness Score: {results_df['fitness'].max():.2f}")
    print(f"Average Fitness (valid): {valid_strategies['fitness'].mean():.2f}")
    print(f"Median Fitness (valid): {valid_strategies['fitness'].median():.2f}")
    
    return top_10, results_df

if __name__ == "__main__":
    # Run the test
    top_strategies, all_results = test_fitness_function(
        pop_size=10000,
        seed=42,
        start_date="1989-12-31", 
        end_date="2020-12-31"
    )