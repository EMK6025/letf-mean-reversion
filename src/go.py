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
from fitness import fitness, calc_metrics

warnings.filterwarnings("ignore", category=FutureWarning, module='vectorbt')
vbt.settings.array_wrapper['freq'] = '1D'

WINDOW_MIN, WINDOW_MAX = 3, 20
ENTRY_MIN, ENTRY_MAX = 0, 50
EXIT_MIN, EXIT_MAX = 50,100
SELL_THRESH_MIN, SELL_THRESH_MAX = 0,100
POS_SIZE_MIN, POS_SIZE_MAX = 0.0, 1.0

creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

toolbox.register("attr_window", random.randint, WINDOW_MIN, WINDOW_MAX)
toolbox.register("attr_entry", random.randint, ENTRY_MIN, ENTRY_MAX)
toolbox.register("attr_exit", random.randint, EXIT_MIN, EXIT_MAX)
toolbox.register("attr_sell_thresh", random.randint, SELL_THRESH_MIN, SELL_THRESH_MAX)
toolbox.register("attr_pos_size", random.uniform, POS_SIZE_MIN, POS_SIZE_MAX)

engine = create_engine()
df = connect(engine, "test_data")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)
spxt = df["SPX Close"]
benchmark = Portfolio.from_holding(close=spxt, freq='1D')
benchmark_returns = benchmark.returns()

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

def evaluate(pop, start_date, end_date, generation=0):
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
        
    pfs = backtest.run(params, start_date, end_date)
    
    sortino, sharpe, rel_drawdown, _, alpha, _ = calc_metrics(pfs, benchmark)
    
    fitness_vals = fitness(sortino, sharpe, rel_drawdown, alpha, generation)
    
    return list(zip(*fitness_vals))

def run_population(pop, start_date="1989-12-31", end_date="2020-12-31", generation=0):
    fitnesses = evaluate(pop, start_date, end_date, generation)
    
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    return pop

def create_initial_population(pop_size=50, start_date="1989-12-31", end_date="2020-12-31"):
    pop = toolbox.population(n=pop_size)

    return pop

def show_population(pop):
    print("\nPopulation Fitness Scores (Multi-Objective):")
    print("="*80)
    print(f"{'#':<3} {'Sortino':<8} {'Sharpe':<8} {'RelDD':<8} {'Alpha':<8} {'Window':<6} {'Entry':<5} {'Exit':<4}")
    print("-"*80)
    
    for i, ind in enumerate(pop):
        sortino, sharpe, rel_dd, alpha = ind.fitness.values
        print(f"{i:<3} {sortino:<8.3f} {sharpe:<8.3f} {rel_dd:<8.3f} {alpha:<8.3f} "
              f"{ind[0]:<6} {ind[1]:<5} {ind[2]:<4}")

def create_next_generation(population, cx_prob=0.5, mut_prob=0.2, start_date="1989-12-31", end_date="2020-12-31", generation=0):
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
    
    offspring = []
    for _ in range(pop_size):
        if random.random() < cx_prob and pop_size >= 2:
            p1, p2 = random.sample(population, 2)
            c1, c2 = toolbox.clone(p1), toolbox.clone(p2)
            toolbox.mate(c1, c2)
            child = random.choice([c1, c2])
            ps = sorted(child[4:15])
            child[4:15] = ps
        else:
            child = toolbox.clone(random.choice(population))
        
        if random.random() < mut_prob:
            toolbox.mutate(child)
            child[0] = max(WINDOW_MIN, min(WINDOW_MAX, int(round(child[0]))))
            child[1] = max(ENTRY_MIN, min(ENTRY_MAX, int(round(child[1]))))
            child[2] = max(EXIT_MIN, min(EXIT_MAX, int(round(child[2]))))
            child[3] = max(SELL_THRESH_MIN, min(SELL_THRESH_MAX, int(round(child[3]))))
            pos_sizes = sorted([max(POS_SIZE_MIN, min(POS_SIZE_MAX, x)) for x in child[4:15]])
            child[4:15] = pos_sizes
        
        del child.fitness.values
        offspring.append(child)
    
    offspring = run_population(offspring, start_date=start_date, end_date=end_date, generation=generation)

    combined = population + offspring
    next_gen = toolbox.select(combined, k=pop_size)
    
    return next_gen

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selNSGA2)