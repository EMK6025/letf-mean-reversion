import pandas as pd
from engine import create_engine, connect_time_series
from vectorbt import Portfolio, settings
import warnings

def wfo():
    from wfo import walk_forward_optimization
    from fitness import FitnessConfig
    start_date = "1991-01-01"
    end_date = "2024-12-31"

    custom_config = FitnessConfig(
        selected_metrics=['sortino', 'sharpe', 'rel_drawdown', 'alpha', 'activeness', 'drawdown', 'var', 'annual_return', 'pain_ratio', 'information_ratio'],
        enable_bottom_percentile_filter=True,
        bottom_percentile=10.0
    )

    walk_forward_optimization(
        start_date=start_date,
        end_date=end_date,  
        in_sample_months=120,
        out_sample_months=12,
        max_time_minutes=20,
        stall_generations=10,
        max_generations=50,
        pop_size=1000,
        n_ensemble=10,
        leverage=2,
        fitness_config=custom_config
    )

    walk_forward_optimization(
        start_date=start_date,
        end_date=end_date,  
        in_sample_months=120,
        out_sample_months=12,
        max_time_minutes=20,
        stall_generations=10,
        max_generations=50,
        pop_size=1000,
        n_ensemble=10,
        leverage=3,
        fitness_config=custom_config
    )

    custom_config = FitnessConfig(
        selected_metrics=['sortino', 'sharpe', 'rel_drawdown', 'alpha', 'activeness', 'var'],
        enable_bottom_percentile_filter=True,
        bottom_percentile=10.0
    )

    walk_forward_optimization(
        start_date=start_date,
        end_date=end_date,  
        in_sample_months=120,
        out_sample_months=12,
        max_time_minutes=20,
        stall_generations=10,
        max_generations=101,
        pop_size=1000,
        n_ensemble=10,
        leverage=3,
        fitness_config=custom_config
    )

def PCA(run_index):
    from backtest_analysis import PCA_analysis
    PCA_analysis(run_index)

def list_runs():
    engine = create_engine()
    run = pd.read_sql(f"SELECT run_id, COUNT(*) AS occurrences FROM wfo_run GROUP BY run_id ORDER BY occurrences DESC;", engine)
    print(run)

def reset_wfo():
    from wfo_sql import reset
    engine = create_engine()
    reset(engine)
    
if __name__ == "__main__":
    wfo()