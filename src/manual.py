from typing import List

def wfo():
    from wfo import walk_forward_optimization
    from fitness import FitnessConfig
    import traceback

    start_date = "1991-01-01"
    end_date = "2024-12-31"

    custom_config = FitnessConfig(
        selected_metrics=['sortino', 'sharpe', 'rel_drawdown', 'alpha', 
                          'activeness', 'drawdown', 'annual_return', 'pain_ratio', 'var', 'information_ratio'],
        enable_bottom_percentile_filter=True,
        bottom_percentile=10.0
    )
    
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
            n_ensemble=10,
            leverage=2,
            fitness_config=custom_config
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
            leverage=2,
            fitness_config=custom_config
        )
    except Exception as e:
        print("fail 2")
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
            n_ensemble=10,
            leverage=3,
            fitness_config=custom_config
        )
    except Exception as e:
        print("fail 3")
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
            fitness_config=custom_config
        )
    except Exception as e:
        print("fail 4")
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
            n_ensemble=10,
            leverage=4,
            fitness_config=custom_config
        )
    except Exception as e:
        print("fail 5")
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
            leverage=4,
            fitness_config=custom_config
        )
    except Exception as e:
        print("fail 6")
        traceback.print_exc()

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
    
if __name__ == "__main__":
    
    # fit()
    # import pandas as pd
    # from engine import create_engine
    # engine = create_engine()
    # run = pd.read_sql(f"SELECT * FROM wfo_run WHERE run_id = 3;", engine)
    # print(run.to_string())
    
    # import pandas as pd
    # from engine import create_engine
    # engine = create_engine()
    # run = pd.read_sql("""
    #     SELECT
    #     period_id,
    #     COUNT(*) AS occurrences
    #     FROM wfo_strategy
    #     WHERE run_id = 6
    #     GROUP BY period_id
    #     ORDER BY period_id ASC
    #     """, engine)
    # print(run)
    
    # import pandas as pd
    # from engine import create_engine
    # engine = create_engine()
    # periods = pd.read_sql(f"SELECT * FROM wfo_period_summary WHERE run_id = 2;", engine)
    # for i in range(0, len(periods)):
    #     print(f"{periods.iloc[i]['in_sample_end']} - {periods.iloc[i]['out_sample_end']}")
        
    # reset_wfo()
    # list_runs()
    
    # clear_run([1])
    # list_runs()
    
    # wfo()   
    run(3)
    run(4)
    # PCA(1)
    # x = input("go next?")
    # PCA(2)
    # x = input("go next?")
    # PCA(3)
    

    # from backtest import Params, run
    # import pandas as pd
    # print("start code")
    # params =  [Params(5, 20, 70, 10, [.2, .6, .8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 
    #            Params(3, 40, 90, 20, [.2, .6, .8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]
    # start_date = "1991-01-01"
    # end_date = "1996-12-30"
    # actual_end_date = pd.to_datetime("1995-12-31")
    # pfs = run(params, start_date=start_date, end_date=end_date, stop_entry_date=actual_end_date, leverage=3)
    # records = pfs.trades.records_readable.copy()
    # records = records[records['Status'] == 'Closed']["Exit Timestamp"].copy()
    # final_exit = records.max()
    # print("last exit after cutoff:", pd.Timestamp(final_exit))
