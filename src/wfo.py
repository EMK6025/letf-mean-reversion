from go import run_initial_population, create_next_generation
from backtest import Portfolio

# Generate and evaluate 500 random strategies
# population = run_initial_population(pop_size=500)


pf = Portfolio.from_orders(
        close      = letf,
        size       = size_changes,
        size_type  = 'targetpercent',
        freq       = '1D'
    )