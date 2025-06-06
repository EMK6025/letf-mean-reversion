import pandas as pd
import backtest
from backtest import Params
from engine import create_engine, connect
from vectorbt import Portfolio
import numpy as np
import matplotlib.pyplot as plt
import vectorbt as vbt
from deap import base, creator, tools
import warnings
import math

warnings.filterwarnings("ignore", category=FutureWarning, module='vectorbt')
vbt.settings.array_wrapper['freq'] = '1D'
SORTINO_THRESHOLD = .5
OMEGA_THRESHOLD = .85
DRAWDOWN_THRESHOLD = .5
ALPHA_SCALING = 15
ALPHA_THRESHOLD = .025

SORTINO_WEIGHT = 30
OMEGA_WEIGHT = 30
ALPHA_WEIGHT = 40

# def fitness(sortino, omega, relative_drawdowns, alpha):
#     if sortino < SORTINO_THRESHOLD or omega < OMEGA_THRESHOLD or relative_drawdowns < DRAWDOWN_THRESHOLD:
#         return (-1000,) # failure, kick immediately
    
#     norm_sortino = (min(sortino, 3) - SORTINO_THRESHOLD) * SORTINO_WEIGHT / (3 - SORTINO_THRESHOLD)
#     norm_omega = (min(omega, 3) - OMEGA_THRESHOLD) * OMEGA_WEIGHT / (3 - OMEGA_THRESHOLD)
#     norm_alpha = 1 / (1 + math.exp(-ALPHA_SCALING * (alpha - ALPHA_THRESHOLD))) * ALPHA_WEIGHT

#     return ((norm_sortino + norm_omega + norm_alpha) * (relative_drawdowns),)

def fitness(sortino, omega, relative_drawdowns, alpha):
    fail_mask = (sortino < SORTINO_THRESHOLD) | (omega < OMEGA_THRESHOLD) | (relative_drawdowns < DRAWDOWN_THRESHOLD)
    
    norm_sortino = (np.minimum(sortino, 3) - SORTINO_THRESHOLD) * SORTINO_WEIGHT / (3 - SORTINO_THRESHOLD)
    norm_omega = (np.minimum(omega, 3) - OMEGA_THRESHOLD) * OMEGA_WEIGHT / (3 - OMEGA_THRESHOLD)
    norm_alpha = 1 / (1 + np.exp(-ALPHA_SCALING * (alpha - ALPHA_THRESHOLD))) * ALPHA_WEIGHT

    fitness = (norm_sortino + norm_omega + norm_alpha) * (relative_drawdowns)
    print("sortino: ", sortino)
    print("omega: ", omega)
    print("relative_drawdowns: ", relative_drawdowns)
    print("alpha: ", alpha)
    print("  ")
    
    fitness[fail_mask] = -1000
    return list(zip(fitness))
    

def temp():
    def evaluate_strategy(individual):
        # Example: individual = [fast_ma, slow_ma, stop_loss]
        params = {
            "fast_ma": individual[0],
            "slow_ma": individual[1],
            "stop_loss": individual[2],
        }
        
        # Run vectorbt backtest
        result = run_vectorbt_backtest(params)
        
        # Extract fitness (maximize Sortino for example)
        fitness = result.sortino_ratio
        
        return (fitness,)  # Must return a tuple!

    # Maximize fitness
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define individual with 3 parameters as floats
    toolbox.register("attr_float", random.uniform, 5, 100)  # Change range as needed
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register evaluation function
    toolbox.register("evaluate", evaluate_strategy)

    # Register genetic operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=50)
    NGEN = 20
    CXPB, MUTPB = 0.5, 0.2

    for gen in range(NGEN):
        # Evaluate fitness
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Select, clone, and apply genetic ops
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the new offspring (only those with invalid fitness)
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_inds)
        for ind, fit in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit

        # Replace old population
        population[:] = offspring

    def init_param_1():
        return random.uniform(3, 20)  # Param 1: float in [3, 20]

    def init_param_2():
        return random.uniform(0, 100)  # Param 2: float in [0, 100]

    def init_param_3_array():
        # Param 3: increasing array of 10 values in [0, 1]
        values = sorted([random.uniform(0, 1.0) for _ in range(10)])
        return values

    from deap import creator, base, tools

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def init_individual():
        p1 = init_param_1()
        p2 = init_param_2()
        p3 = init_param_3_array()
        return creator.Individual([p1, p2, p3])

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def custom_mutate(individual, indpb=0.2):
        # Mutate param 1 (index 0): float between 3 and 20
        if random.random() < indpb:
            individual[0] = min(20, max(3, individual[0] + random.gauss(0, 1)))

        # Mutate param 2 (index 1): float between 0 and 100
        if random.random() < indpb:
            individual[1] = min(100, max(0, individual[1] + random.gauss(0, 5)))

        # Mutate param 3 (index 2): 10-length increasing array in [0, 1]
        if random.random() < indpb:
            new_array = individual[2][:]
            for i in range(len(new_array)):
                if random.random() < 0.5:
                    change = random.gauss(0, 0.05)
                    new_array[i] = min(1.0, max(0.0, new_array[i] + change))
            new_array = sorted(new_array)
            individual[2] = new_array

        return (individual,)

    toolbox.register("mutate", custom_mutate)


    import random
    from deap import tools

    # Parameters
    POP_SIZE = 50
    NGEN = 20
    CXPB = 0.5  # Crossover probability
    MUTPB = 0.2  # Mutation probability

    # Initialize population
    population = toolbox.population(n=POP_SIZE)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Start evolution
    for gen in range(NGEN):
        print(f"-- Generation {gen} --")

        # Select next generation individuals
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Re-evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace old population
        population[:] = offspring

        # Log best individual
        top1 = tools.selBest(population, 1)[0]
        print("Best individual:", top1)
        print("Fitness:", top1.fitness.values[0])
