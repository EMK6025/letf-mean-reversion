import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vectorbt import Portfolio
from datetime import date
from engine import create_engine, connect, connect_time_series
from vectorbt import settings
settings.array_wrapper['freq'] = '1D'

def rebuild_performance(run_id):
    from backtest import run, Params
    engine = create_engine()
    runs = pd.read_sql(f"SELECT * FROM wfo_run WHERE run_id = {run_id} LIMIT 1", engine)
    periods = pd.read_sql(f"SELECT * FROM wfo_period_summary WHERE run_id = {run_id}", engine)
    
    strategies = connect(engine, "wfo_strategy")
    strategies = (
        strategies
        [strategies['run_id'] == run_id]
        .sort_values(by='period_id')
        .reset_index(drop=True)
    )    
    
    # print(f"length of strategies: {len(strategies)}")
    start_date = pd.to_datetime(runs['start_date'].iloc[0])
    end_date = pd.to_datetime(runs['end_date'].iloc[0])
    in_sample_months = runs['in_sample_months'].iloc[0]
    leverage = runs['leverage'].iloc[0]
        
    # take start date, then offset by first in-sample period plus 1 day
    backtest_start_date = start_date + pd.DateOffset(months=in_sample_months) + pd.DateOffset(days=1)
    
    # for col in ['pos_sizing', 'fitness_values']:
    #     strategies[col] = strategies[col].apply(lambda x: [round(v, 2) for v in x])

    df = connect_time_series(engine, "test_data")
    spx_after_date = df["SPX Close"][backtest_start_date:].iloc[0]

    current_portfolio_value = spx_after_date
    
    cumulative_values = pd.DataFrame()
       
    run_period = 1
    
    while True:
        current_period = periods[periods['period_index'] == run_period].iloc[0]
        period_id = current_period['period_id']
        
        period_start_date = current_period['in_sample_end'] + pd.DateOffset(days=1)
        period_end_date = pd.Timestamp(current_period['out_sample_end'])
        if period_end_date > end_date:
            period_end_date = end_date
            
        period_strategies = strategies[strategies['period_id'] == period_id]
        if len(period_strategies) == 0:
            break        
        
        period_ensemble_params = []
        for i in range(0,len(period_strategies)):
            period_ensemble_params.append(Params(
                window=int(period_strategies["window"].iloc[i]),
                entry=int(period_strategies["entry"].iloc[i]),
                exit=int(period_strategies["exit"].iloc[i]),
                sell_threshold=int(period_strategies["sell_threshold"].iloc[i]),
                position_sizing=np.array([float(x) for x in period_strategies["pos_sizing"].iloc[i]])
            ))
        
        initial_capital_per_strategy = current_portfolio_value/len(period_ensemble_params)
        
        pfs = run(period_ensemble_params, period_start_date, 
                  period_end_date, period_end_date, 
                  initial_capital_per_strategy, leverage)
        
        combined_performance = pfs.value().sum(axis=1)
        
        pfs = Portfolio.from_holding(
            close = combined_performance.loc[:end_date],
            size  = 1,
            freq  = pfs.wrapper.freq
        )
                
        if not cumulative_values.empty:
            cumulative_values = pd.concat([cumulative_values, combined_performance], axis=0)
        else:
            cumulative_values = combined_performance.copy()
   
        print(f"Period {run_period}: Start Date {period_start_date}")
        print(f"                 End Date {period_end_date}")
        print(f"           Starting value {current_portfolio_value}")
        print(f"             Ending value {combined_performance.iloc[-1]}")
        
        current_portfolio_value = combined_performance.iloc[-1]
        period_start_date = period_end_date + pd.DateOffset(days=1)
        if period_start_date > end_date:
            break
        
        run_period += 1
        
    return cumulative_values, backtest_start_date, leverage

def analyze_wfo(run_id):
    """
    Analyze walk-forward optimization results with ensemble strategies.
    """
    cumulative_values, backtest_start_date, leverage = rebuild_performance(run_id)
    engine = create_engine()
    df = connect_time_series(engine, "test_data")
    spx = df["SPX Close"][backtest_start_date:]

    pfs = Portfolio.from_holding(
        close=cumulative_values,
        size = 1,
        freq='1D'
    )
    
    benchmark = Portfolio.from_holding(
        close=spx, 
        size=1, 
        freq=pfs.wrapper.freq,
    )

    returns = pfs.returns()
    benchmark_returns = benchmark.annualized_return()
    sortino = returns.vbt.returns.sortino_ratio()
    sharpe = returns.vbt.returns.sharpe_ratio()
    rel_drawdown = (pfs.max_drawdown() / benchmark.max_drawdown())
    drawdown = pfs.max_drawdown()
    alpha = (pfs.annualized_return() - benchmark.annualized_return())
    annual_return = pfs.annualized_return()
    
    print("=== Performance Metrics ===")
    print(f"Annualized Return       : {annual_return:.2%}")
    print(f"Alpha                   : {alpha:.2%}")
    print(f"Sharpe Ratio            : {sharpe:.2f}")
    print(f"Sortino Ratio           : {sortino:.2f}")
    print(f"Max Drawdown            : {drawdown:.2%}")
    print(f"Relative Drawdown (vs benchmark): {rel_drawdown:.2f}×")
    print(f"Benchmark Annualized Return: {benchmark_returns:.2%}")
    
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(spx.index, spx.values, 
            label='SPX', color='red', linewidth=2, alpha=0.7)
    ax.plot(cumulative_values.index, cumulative_values.values, 
            label='WFO Ensemble Strategy', color='green', linewidth=2)
    
    ax.set_title('Walk-Forward Optimization Ensemble Performance (Continuous Out-of-Sample)', fontsize=14)
    ax.set_ylabel('Portfolio Value')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig("wfo_analysis.png", dpi=300)  # Save the figure with high resolution
    plt.show()

def analyze_PCA(run_id):
    from sklearn.decomposition import PCA

    from sklearn import preprocessing
    
    engine = create_engine()
    run = pd.read_sql(f"SELECT * FROM wfo_run WHERE run_id = {run_id} LIMIT 1", engine)
    strategies = connect(engine, "wfo_strategy")
    strategies = strategies[strategies['run_id'] == run_id]
    
    # grab run.fitness config
    fitness = strategies['fitness_values']
    metrics = run['fitness_config'].iloc[0]['selected_metrics']
    df = pd.DataFrame(fitness.tolist(), columns=metrics)
    
    print(df.head())
    print(df.shape)
    scaled_df = preprocessing.scale(df) #if column samples, pass in df.T

    pca = PCA()
    pca.fit(scaled_df)
    pca_data = pca.transform(scaled_df)

    # #matplotlib scree plot
    per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    plt.bar(x=range(1, len(per_var)+1), height=per_var)
    plt.xticks(ticks=range(1, len(per_var)+1), labels=labels)

    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

    pca_df = pd.DataFrame(pca_data, columns=labels)


    # plt.title('My PCA Graph')
    # plt.scatter(pca_df.PC1, pca_df.PC2)
    
    # plt.xlabel('PC1 = {0}%'.format(per_var[0]))
    # plt.ylabel('PC2 = {0}%'.format(per_var[1]))
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(pca_df.PC1, pca_df.PC2, pca_df.PC3)
    ax.set_xlabel('PC1 = {0}%'.format(per_var[0]))
    ax.set_ylabel('PC2 = {0}%'.format(per_var[1]))
    ax.set_zlabel('PC3 = {0}%'.format(per_var[2]))
    ax.set_title('My PCA Graph')
    plt.show()
    
    loading_scores = pd.Series(pca.components_[0], index=df.keys())
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    top_X_features = sorted_loading_scores[0:8].index.values
    print(loading_scores[top_X_features])

    print(pca.explained_variance_)        # the eigenvalues (variances explained by each PC)
    print(pca.explained_variance_ratio_)  # fraction of total variance per PC
    # print(pca.components_)
    
def analyze_fit():
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
    