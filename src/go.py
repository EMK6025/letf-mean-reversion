import pandas as pd
import backtest
from backtest import Params
from engine import create_engine, connect
from vectorbt import Portfolio
import numpy as np
import matplotlib.pyplot as plt
import vectorbt as vbt
from deap import base, creator, tools
import random
import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning, module='vectorbt')
vbt.settings.array_wrapper['freq'] = '1D'
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

# Parameter ranges for randomization
WINDOW_MIN, WINDOW_MAX = 3, 20
ENTRY_MIN, ENTRY_MAX = 0, 50
EXIT_MIN, EXIT_MAX = 50,100
SELL_THRESH_MIN, SELL_THRESH_MAX = 0,100
POS_SIZE_MIN, POS_SIZE_MAX = 0.0, 1.0

# DEAP setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generators for randomization
toolbox.register("attr_window", random.randint, WINDOW_MIN, WINDOW_MAX)
toolbox.register("attr_entry", random.randint, ENTRY_MIN, ENTRY_MAX)
toolbox.register("attr_exit", random.randint, EXIT_MIN, EXIT_MAX)
toolbox.register("attr_sell_thresh", random.randint, SELL_THRESH_MIN, SELL_THRESH_MAX)
toolbox.register("attr_pos_size", random.uniform, POS_SIZE_MIN, POS_SIZE_MAX)

# Load data once for efficiency
engine = create_engine()
df = connect(engine, "test_data")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)
spxt = df["SPX Close"]
benchmark = Portfolio.from_holding(close=spxt, freq='1D')
benchmark_returns = benchmark.returns()

# Function to create a single individual (strategy parameters)
def create_individual():
    ind = [
        toolbox.attr_window(),
        toolbox.attr_entry(),
        toolbox.attr_exit(),
        toolbox.attr_sell_thresh(),
    ]
    pos = sorted([toolbox.attr_pos_size() for _ in range(11)])
    ind.extend(pos)
    return ind

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

def omega_ratio(returns, benchmark):
    excess_returns = returns.sub(benchmark, axis=0)
    gains  = excess_returns.clip(lower=0)
    losses = excess_returns.clip(upper=0)
    mean_gains = gains.mean(axis=0)
    mean_losses = losses.abs().mean(axis=0)
    if isinstance(mean_losses, pd.Series):
        mean_losses = mean_losses.replace(0, pd.NA)
    else:
        mean_losses = float(mean_losses) if mean_losses != 0 else float("nan")
    return mean_gains / mean_losses


# Existing fitness function
def fitness(sortino, sharpe, omega, relative_drawdowns, alpha):
    fail_mask = (sortino < SORTINO_THRESHOLD) | (sharpe < SHARPE_THRESHOLD) | (omega < OMEGA_THRESHOLD) | (relative_drawdowns < DRAWDOWN_THRESHOLD)
    
    norm_sortino = (np.minimum(sortino, 3) - SORTINO_THRESHOLD) * SORTINO_WEIGHT / (3 - SORTINO_THRESHOLD)
    norm_sharpe = (np.minimum(sharpe, 3) - SHARPE_THRESHOLD) * SHARPE_WEIGHT / (3 - SHARPE_THRESHOLD)
    norm_omega = (np.minimum(omega, 3) - OMEGA_THRESHOLD) * OMEGA_WEIGHT / (3 - OMEGA_THRESHOLD)
    norm_alpha = 1 / (1 + np.exp(-ALPHA_SCALING * (alpha - ALPHA_THRESHOLD))) * ALPHA_WEIGHT
    norm_drawdown = 1 / (1 + np.exp(-DRAWDOWN_SCALING * (relative_drawdowns - DRAWDOWN_THRESHOLD)))

    fitness_val = (norm_sharpe + norm_omega + norm_alpha) * (norm_drawdown)
    
    fitness_val[fail_mask] = -1000
    return list(zip(fitness_val))

# # Evaluate a population
def evaluate(pop, start_date, end_date):
    # Params conversion
    params = []
    for individual in pop:
        window, entry, exit_, sell_threshold, *pos_sizing = individual
        params.append(Params(
            window=window,
            entry=entry,
            exit=exit_,
            sell_threshold=sell_threshold,
            position_sizing=np.array(pos_sizing)
        ))
    # Backtest
    pfs = backtest.run(params, start_date, end_date)
    
    # Metrics
    returns = pfs.returns()
    sortino  = returns.vbt.returns.sortino_ratio()
    sharpe   = returns.vbt.returns.sharpe_ratio()
    omega    = omega_ratio(returns, benchmark_returns)
    drawdown = benchmark.max_drawdown() / pfs.max_drawdown()
    alpha    = pfs.annualized_return() - benchmark.annualized_return()
    
    # Fitness
    fitness_vals = fitness(sortino, sharpe, omega, drawdown, alpha)    
    return fitness_vals


# Function to generate initial population, run backtests, and print fitness scores
def create_initial_population(pop_size=50, start_date="1989-12-31", end_date="2020-12-31"):
    
    # Generate population
    pop = toolbox.population(n=pop_size)

    return pop


def run_population(pop, start_date="1989-12-31", end_date="2020-12-31"):
    # Evaluate fitness for each individual
    fitnesses = evaluate(pop, start_date, end_date)
    
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    return pop
    
        

def show_population(pop):
    # Print results
    print("\nInitial Population Fitness Scores:")
    for i, ind in enumerate(pop):
        print(f"Strategy {i}: Fitness = {ind.fitness.values[0]:.4f}")
        print(f"  Params: window={ind[0]}, entry={ind[1]}, exit={ind[2]}, sell_threshold={ind[3]}")
        print(f"  Position Sizing: {[round(x, 3) for x in ind[4:15]]}\n")


def create_next_generation(population, cx_prob=0.5, mut_prob=0.2):
    pop_size = len(population)
    """
    Takes the current population and their fitness scores, then generates the next
    generation using selection, crossover, and mutation.
    
    Args:
        population: List of DEAP individuals with fitness values
        cx_prob: Probability of crossover (default 0.5)
        mut_prob: Probability of mutation (default 0.2)
    
    Returns:
        List of individuals representing the next generation
    """
    
    # Select parents and clone them to create offspring
    valid = [ind for ind in population if ind.fitness.values[0] > -1000]
    survivors_count = min(pop_size//2, len(valid))
    offspring = tools.selBest(valid, k=survivors_count) 
    offspring = list(map(toolbox.clone, offspring))
    
    while len(offspring) < pop_size:
        p1, p2 = random.sample(offspring, 2)
        c1, c2 = toolbox.clone(p1), toolbox.clone(p2)
        if random.random() < cx_prob:
            toolbox.mate(c1, c2)
            for child in (c1, c2):
                ps = sorted(child[4:15])
                child[4:15] = ps

        del c1.fitness.values
        del c2.fitness.values

        offspring.extend([c1, c2])
            

    offspring = offspring[:pop_size]

    
    # Apply mutation and fix position sizing
    for mutant in offspring:
        if random.random() < mut_prob:
            toolbox.mutate(mutant)
            # Clamp values to valid ranges
            mutant[0] = max(WINDOW_MIN, min(WINDOW_MAX, int(round(mutant[0]))))
            mutant[1] = max(ENTRY_MIN, min(ENTRY_MAX, int(round(mutant[1]))))
            mutant[2] = max(EXIT_MIN, min(EXIT_MAX, int(round(mutant[2]))))
            mutant[3] = max(SELL_THRESH_MIN, min(SELL_THRESH_MAX, int(round(mutant[3]))))
            
            # Fix position sizing to maintain non-decreasing order
            pos_sizes = [max(POS_SIZE_MIN, min(POS_SIZE_MAX, mutant[i])) for i in range(4, 15)]
            pos_sizes.sort()  # Ensure non-decreasing order
            mutant[4:15] = pos_sizes
            
            del mutant.fitness.values
    
    return offspring