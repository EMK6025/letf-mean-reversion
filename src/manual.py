from typing import List

def wfo():
    from wfo import walk_forward_optimization
    from fitness import FitnessConfig
    import traceback
    start_date = "1991-01-01"
    end_date = "2024-12-31"
    
    custom_config = FitnessConfig(
        selected_metrics=['sortino', 'drawdown', 'annual_return', 'var'],
        enable_bottom_percentile_filter=True,
        bottom_percentile=10.0
    )
    
    # try:
    #     walk_forward_optimization(
    #         start_date=start_date,
    #         end_date=end_date,  
    #         in_sample_months=60,
    #         out_sample_months=6,
    #         max_time_minutes=1000,
    #         stall_generations=10,
    #         max_generations=1000,
    #         pop_size=1000,
    #         n_ensemble=50,
    #         leverage=3,
    #         fitness_config=custom_config,
    #         rand_seed = 123
    #     )
    # except Exception as e:
    #     print("fail 1")
    #     traceback.print_exc()

    # try:
    #     walk_forward_optimization(
    #         start_date=start_date,
    #         end_date=end_date,  
    #         in_sample_months=60,
    #         out_sample_months=6,
    #         max_time_minutes=1000,
    #         stall_generations=10,
    #         max_generations=1000,
    #         pop_size=1000,
    #         n_ensemble=50,
    #         leverage=3,
    #         fitness_config=custom_config,
    #         rand_seed = 1234
    #     )
    # except Exception as e:
    #     print("fail 2")
    #     traceback.print_exc()
        
    # try:
    #     walk_forward_optimization(
    #         start_date=start_date,
    #         end_date=end_date,  
    #         in_sample_months=60,
    #         out_sample_months=6,
    #         max_time_minutes=1000,
    #         stall_generations=10,
    #         max_generations=1000,
    #         pop_size=1000,
    #         n_ensemble=50,
    #         leverage=3,
    #         fitness_config=custom_config,
    #         rand_seed = 12345
    #     )
    # except Exception as e:
    #     print("fail 3")
    #     traceback.print_exc()
        
    # try:
    #     walk_forward_optimization(
    #         start_date=start_date,
    #         end_date=end_date,  
    #         in_sample_months=60,
    #         out_sample_months=6,
    #         max_time_minutes=1000,
    #         stall_generations=10,
    #         max_generations=1000,
    #         pop_size=1000,
    #         n_ensemble=50,
    #         leverage=3,
    #         fitness_config=custom_config,
    #         rand_seed = 234
    #     )
    # except Exception as e:
    #     print("fail 1")
    #     traceback.print_exc()

    # try:
    #     walk_forward_optimization(
    #         start_date=start_date,
    #         end_date=end_date,  
    #         in_sample_months=60,
    #         out_sample_months=6,
    #         max_time_minutes=1000,
    #         stall_generations=10,
    #         max_generations=1000,
    #         pop_size=1000,
    #         n_ensemble=50,
    #         leverage=3,
    #         fitness_config=custom_config,
    #         rand_seed = 2345
    #     )
    # except Exception as e:
    #     print("fail 2")
    #     traceback.print_exc()
        
    # try:
    #     walk_forward_optimization(
    #         start_date=start_date,
    #         end_date=end_date,  
    #         in_sample_months=60,
    #         out_sample_months=6,
    #         max_time_minutes=1000,
    #         stall_generations=10,
    #         max_generations=1000,
    #         pop_size=1000,
    #         n_ensemble=50,
    #         leverage=3,
    #         fitness_config=custom_config,
    #         rand_seed = 23456
    #     )
    # except Exception as e:
    #     print("fail 3")
    #     traceback.print_exc()

    try:
        walk_forward_optimization(
            start_date=start_date,
            end_date=end_date,  
            in_sample_months=60,
            out_sample_months=6,
            max_time_minutes=1000,
            stall_generations=10,
            max_generations=1000,
            pop_size=1000,
            n_ensemble=50,
            leverage=3,
            fitness_config=custom_config,
            rand_seed = 5
        )
    except Exception as e:
        print("fail 1")
        traceback.print_exc()

    try:
        walk_forward_optimization(
            start_date=start_date,
            end_date=end_date,  
            in_sample_months=60,
            out_sample_months=6,
            max_time_minutes=1000,
            stall_generations=10,
            max_generations=1000,
            pop_size=1000,
            n_ensemble=50,
            leverage=3,
            fitness_config=custom_config,
            rand_seed = 13
        )
    except Exception as e:
        print("fail 2")
        traceback.print_exc()
        
    # try:
    #     walk_forward_optimization(
    #         start_date=start_date,
    #         end_date=end_date,  
    #         in_sample_months=60,
    #         out_sample_months=6,
    #         max_time_minutes=1000,
    #         stall_generations=10,
    #         max_generations=1000,
    #         pop_size=1000,
    #         n_ensemble=50,
    #         leverage=3,
    #         fitness_config=custom_config,
    #         rand_seed = 34567
    #     )
    # except Exception as e:
    #     print("fail 3")
    #     traceback.print_exc()
    
def PCA(run_index):
    from backtest_analysis import analyze_PCA
    analyze_PCA(run_index)

def run(run_id):
    from backtest_analysis import analyze_wfo
    analyze_wfo(run_id)

def fit():
    from backtest_analysis import analyze_fit
    analyze_fit()
    
def list_runs():
    import pandas as pd
    from engine import create_engine
    engine = create_engine()
    run = pd.read_sql(f"SELECT run_id FROM wfo_run ORDER BY run_id ASC;", engine)
    print(run)
    
def reset_wfo():
    from engine import create_engine
    from wfo_sql import reset
    engine = create_engine()
    reset(engine)
    
def clear_runs(to_remove: List):
    from engine import create_engine
    from wfo_sql import clear
    engine = create_engine()
    clear(engine, to_remove)
    
def alpha_all():
    from backtest_analysis import analyze_alpha_all
    analyze_alpha_all()

def alpha():
    from backtest_analysis import analyze_probability_of_outperformance
    analyze_probability_of_outperformance()
    
if __name__ == "__main__":
    # import pandas as pd
    # from engine import create_engine
    # engine = create_engine()
    # with pd.option_context(
    #     'display.max_rows', None,
    #     'display.max_columns', None,
    #     'display.width', None,
    #     'display.max_colwidth', None
    # ):
    #     run = pd.read_sql(f"SELECT * FROM wfo_run WHERE run_id = 23;", engine)
    #     print(run)

    alpha_all()
    # x = [35, 36, 37, 38, 40, 41, 43, 44, 45]
    # from backtest_analysis import analyse_rsi
    # analyse_rsi(x)
    # from backtest_analysis import analyze_probability_of_outperformance
    # analyze_probability_of_outperformance(x)
    # wfo()
    # clear_runs([i for i in range(0,35)])
    # list_runs()