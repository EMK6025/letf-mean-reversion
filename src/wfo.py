import go
from backtest import Portfolio

# Generate and evaluate 500 random strategies
population = go.create_initial_population(pop_size=500)

population = go.run_population(population, start_date="1989-12-31", end_date="2020-12-31")

valid = [ind for ind in population if ind.fitness.values[0] > -1000]

print(f"valid strategies: {len(valid)}")