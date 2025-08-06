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

def PCA(run_index):
    from backtest_analysis import analyze_PCA
    analyze_PCA(run_index)

def run(run_id):
    from backtest_analysis import analyze_wfo
    analyze_wfo(run_id)

def fit():
    import numpy as np
    import pandas as pd
    from engine import create_engine, connect
    import matplotlib.pyplot as plt
    engine = create_engine()
    run_ids = pd.read_sql(f"SELECT * FROM wfo_run ORDER BY run_id ASC;", engine)
    all_periods = pd.read_sql(f"SELECT * from wfo_period_summary ORDER BY run_id ASC;", engine)
    all_strategies = pd.read_sql(f"SELECT * from wfo_strategy;", engine)
    gen_to_volume = []
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for row in run_ids.itertuples(index=False):
        run     = row.run_id
        metrics = row.fitness_config["selected_metrics"]
        run_periods = all_periods[all_periods["run_id"] == run]
        sorted_alpha = []
        for period in run_periods['period_id']:
            period_strategies = all_strategies[(all_strategies['run_id'] == run) & (all_strategies['period_id'] == period)]
            fitness = period_strategies['fitness_values']
            fitness = pd.DataFrame(fitness.tolist(), columns=metrics)
            sorted_alpha.append(round(fitness['alpha'].mean(), 2))
        generations = run_periods["generation_count"]
        hypervolume = run_periods["final_hypervolume"]
        ax.scatter(generations, hypervolume, sorted_alpha)
        # grab alpha
        gen_to_volume.extend(list(zip(generations, hypervolume, sorted_alpha)))

    ax.set_title('Generations v.s. Hypervolume')        
    ax.set_xlabel('Generations')
    ax.set_ylabel('Hypervolume')
    ax.set_zlabel('Alpha')
    plt.show()
    
    from sklearn.decomposition import PCA

    from sklearn import preprocessing
    df = pd.DataFrame(gen_to_volume, columns=["Generations", "Hypervolume", "Alpha"])
    scaled_df = preprocessing.scale(df) #if column samples, pass in df.T

    pca = PCA()
    pca.fit(scaled_df)
    pca_data = pca.transform(scaled_df)
    loading_scores = pd.Series(pca.components_[0], index=df.keys())
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    top_X_features = sorted_loading_scores[0:8].index.values
    print(loading_scores[top_X_features])

    # print(pca.explained_variance_)        # the eigenvalues (variances explained by each PC)
    print(pca.explained_variance_ratio_)
    
    
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
    
    fit()
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
    # run = pd.read_sql(f"SELECT * FROM wfo_period_summary WHERE run_id = 1;", engine)
    # print(run.iloc[2].to_string())
        
    # reset_wfo()
    # list_runs()
    
    # clear_run([1])
    # list_runs()
    
    # wfo()
        
    # run(2)
    # x = input("go next")
    # analyze_run(7)
    
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
