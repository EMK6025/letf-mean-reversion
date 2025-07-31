from typing import List

def wfo():
    from wfo import walk_forward_optimization
    from fitness import FitnessConfig
    start_date = "1991-01-01"
    end_date = "2024-12-31"

    custom_config = FitnessConfig(
        selected_metrics=['rel_drawdown', 'alpha', 'activeness', 'pain_ratio', 'var'],
        enable_bottom_percentile_filter=True,
        bottom_percentile=10.0
    )

    walk_forward_optimization(
        start_date=start_date,
        end_date=end_date,  
        in_sample_months=60,
        out_sample_months=6,
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
        in_sample_months=60,
        out_sample_months=6,
        max_time_minutes=20,
        stall_generations=10,
        max_generations=100,
        pop_size=1000,
        n_ensemble=10,
        leverage=2,
        fitness_config=custom_config
    )


def PCA(run_index):
    from backtest_analysis import PCA_analysis
    PCA_analysis(run_index)

def analyze_run(run_id):
    from backtest_analysis import analyze_wfo
    analyze_wfo(run_id)

def list_runs():
    import pandas as pd
    from engine import create_engine
    engine = create_engine()
    run = pd.read_sql(f"SELECT run_id, COUNT(*) AS occurrences FROM wfo_run GROUP BY run_id ORDER BY occurrences DESC;", engine)
    print(run)

def reset_wfo():
    from engine import create_engine
    from wfo_sql import reset
    engine = create_engine()
    reset(engine)
    
def clear_run(to_remove: List):
    from engine import create_engine
    from wfo_sql import clear
    engine = create_engine()
    clear(engine, to_remove)

if __name__ == "__main__":
    
    wfo()
    
    # reset_wfo()
    # list_runs()
    
    # PCA(1)
    # x = input("go next?")
    # PCA(2)
    # x = input("go next?")
    # PCA(3)

    # clear_run([5])
    # list_runs()
