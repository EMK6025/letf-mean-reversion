from typing import List
import random

def wfo():
    from wfo import walk_forward_optimization
    from fitness import FitnessConfig
    import traceback
    
    '''
    Set your parameters
    '''
    start_date = '1990-01-01'   # when the first in-sample period should start
    end_date = '2024-12-31'     # when the last out-of-sample period should end
    random.seed(2025)           # random seed 
    run_count = 5               # how many seeds to backtest on
    in_sample_months = 60       # length of the in-sample period per window 
    out_sample_months = 6       # length of the out-of-sample period per window
    max_time_minutes = 10       # hard cap on how long processing for each window will take
    stall_generations = 10      # number of generations without progress to end the training period
    max_generations = 1000      # maximum number of generations for each window
    pop_size = 1000             # population size per generation
    n_ensemble = 50             # size of the ensemble strategy to test
    leverage = 3                # leverage to use, integer from 1 to 4
    
    custom_config = FitnessConfig(
        selected_metrics=['sortino', 'drawdown', 'annual_return', 'var'],
        enable_bottom_percentile_filter=True,
        bottom_percentile=10.0
    )
    
    for _ in range(0, run_count):
        try:
            walk_forward_optimization(
                start_date=start_date,
                end_date=end_date,  
                in_sample_months=in_sample_months,
                out_sample_months=out_sample_months,
                max_time_minutes=max_time_minutes,
                stall_generations=stall_generations,
                max_generations=max_generations,
                pop_size=pop_size,
                n_ensemble=n_ensemble,
                leverage=leverage,
                fitness_config=custom_config,
                rand_seed = random.randint(1, 10000)
            )
        except Exception as e:
            print('Something went wrong, go fix.')
            traceback.print_exc()
      
def grab_runs():
    from backtest_analysis import list_run_ids
    return list_run_ids().tolist()

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
    wfo()