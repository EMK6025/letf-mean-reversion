# src/fitness.py
import numpy as np
import vectorbt as vbt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module='vectorbt')
vbt.settings.array_wrapper['freq'] = '1D'

def calc_metrics(pfs, benchmark):
    returns = pfs.returns()
    sortino = returns.vbt.returns.sortino_ratio()
    sharpe = returns.vbt.returns.sharpe_ratio()
    rel_drawdown = pfs.max_drawdown() / benchmark.max_drawdown()
    drawdown = pfs.max_drawdown()
    alpha = pfs.annualized_return() - benchmark.annualized_return()
    annual_return = pfs.annualized_return()
    return sortino, sharpe, rel_drawdown, drawdown, alpha, annual_return

def fitness(sortino, sharpe, rel_drawdown, alpha):
    """
    removes non-pareto front individuals in the bottom 10% performers in any metric
    """
    sort_arr = sortino.values
    shar_arr = sharpe.values
    dd_arr   = rel_drawdown.values
    alpha_arr= alpha.values
    def dominates(i, j):
        # Reward metrics: higher is better; drawdown: lower is better
        better_or_eq = (
            sort_arr[i] >= sort_arr[j] and
            shar_arr[i] >= shar_arr[j] and
            dd_arr[i] <= dd_arr[j] and
            alpha_arr[i] >= alpha_arr[j]
        )
        strictly_better = (
            sort_arr[i] > sort_arr[j] or
            shar_arr[i] > shar_arr[j] or
            dd_arr[i] < dd_arr[j] or
            alpha_arr[i] > alpha_arr[j]
        )
        return better_or_eq and strictly_better

    valid_mask = (
        np.isfinite(sortino) & 
        np.isfinite(sharpe) & 
        (rel_drawdown != 0) &
        (alpha > 0)
    )
    
    sortino_10th = np.percentile(sortino[valid_mask], 10)
    sharpe_10th = np.percentile(sharpe[valid_mask], 10)
    alpha_10th = np.percentile(alpha[valid_mask], 10)
    rel_dd_90th = np.percentile(rel_drawdown[valid_mask], 90)  # 90th percentile for drawdown (higher is worse)
    
    bottom_10_mask = (
        (sortino < sortino_10th) |
        (sharpe < sharpe_10th) |
        (rel_drawdown > rel_dd_90th) |
        (alpha < alpha_10th)
    )
    
    n = len(sortino)
    
    pareto_mask  = np.ones(len(sortino), dtype=bool)

    for j in range(n):
        for i in range(n):
            if i != j and dominates(i, j):
                pareto_mask[j] = False
                break
        
    # Combine validity and bottom 10% filtering
    final_mask = valid_mask & (~bottom_10_mask | pareto_mask)
    
    fitness_sortino = np.where(final_mask, sortino, -1000)
    fitness_sharpe = np.where(final_mask, sharpe, -1000)
    fitness_rel_drawdown = np.where(final_mask, rel_drawdown, 1000)
    fitness_alpha = np.where(final_mask, alpha, -1000)

    return (fitness_sortino, fitness_sharpe, fitness_rel_drawdown, fitness_alpha)