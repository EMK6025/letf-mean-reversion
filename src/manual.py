from typing import List

def wfo():
    from wfo import walk_forward_optimization
    from fitness import FitnessConfig
    import traceback
    start_date = '1990-01-01'
    end_date = '2024-12-31'
    
    custom_config = FitnessConfig(
        selected_metrics=['sortino', 'drawdown', 'annual_return', 'var'],
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
            n_ensemble=50,
            leverage=3,
            fitness_config=custom_config,
            rand_seed = 1013
        )
    except Exception as e:
        print('Something when wrong, go fix.')
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
    
def grab_runs():
    from backtest_analysis import list_run_ids
    return list_run_ids().tolist()

def list_runs():
    from backtest_analysis import list_run_ids
    runs = list_run_ids()
    print(run)
    return list_run_ids().tolist()
    
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

def alpha(run_ids):
    from backtest_analysis import analyze_probability_of_outperformance
    analyze_probability_of_outperformance(run_ids)

def graph_base_holdings(run_ids):
    from backtest_analysis import analyse_gameplan
    import matplotlib.pyplot as plt
    plt.figure()
    df = analyse_gameplan(run_ids)
    # groupby returns (run, group)
    for _, group in df.groupby('run'):
        plt.plot(group['dt'], group['avg_base_hold'], alpha=0.35, linewidth=1)
    
    df_total = df.groupby('dt', as_index=False)['avg_base_hold'].mean()
    plt.plot(df_total['dt'], df_total['avg_base_hold'], color='black', linewidth=1.0, label='Mean')

    plt.ylabel('Average Base Allocation')
    plt.xlabel('Date')
    plt.title('Average OOS Base Allocation over Time')
    plt.legend()    
    plt.grid(True)
    plt.show()
    
def graph_performance(run_ids):
    from backtest_analysis import gen_performance
    from engine import create_engine, connect_time_series
    import matplotlib.pyplot as plt
    df = gen_performance(run_ids)
    
    engine = create_engine()
    data = connect_time_series(engine, 'test_data')
        
    spx = data['SPX Close'].rename('SPX').reindex(df.index).dropna()
    
    plt.figure()
    for col in df:
        plt.plot(df.index, df[col], alpha=0.35, linewidth=1)

    plt.plot(df.index, spx, alpha=1.0, color='black', linewidth=1.0, label = 'SPX')
    
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.title('OOS Performance versus SPX')
    plt.legend()    
    plt.grid(True)
    plt.show()
    
if __name__ == '__main__':
    list_runs()