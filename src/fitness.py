# src/fitness.py
import numpy as np
import vectorbt as vbt
import warnings
from pandas import Series
from dataclasses import dataclass
from typing import List, Dict

warnings.filterwarnings("ignore", category=FutureWarning, module='vectorbt')
vbt.settings.array_wrapper['freq'] = '1D'

@dataclass
class FitnessConfig:
    """Configuration for fitness function metrics and weights"""
    
    # metric_name: (weight, minimize boolean, reference point)
    available_metrics = {
        'sortino': (1.0, False, -1.0),           # Higher is better
        'sharpe': (1.0, False, -1.0),            # Higher is better  
        'rel_drawdown': (-1.0, True, 2.0),      # Lower is better
        'alpha': (1.0, False, -.5),             # Higher is better
        'position_stability': (-1.0, True, 1.0), # Lower is better
        'drawdown': (-1.0, True, -1.0),          # Lower is better
        'annual_return': (1.0, False, -1.0)      # Higher is better
    }
    
    selected_metrics: List[str] = None
    
    custom_weights: Dict[str, float] = None
    
    enable_bottom_percentile_filter: bool = True
    bottom_percentile: float = 10.0
    top_percentile: float = 90.0  # for metrics where higher is worse
    
    def __post_init__(self):
        if self.selected_metrics is None:
            # default
            self.selected_metrics = ['sortino', 'sharpe', 'rel_drawdown', 'alpha', 'position_stability']
        
        if self.custom_weights is None:
            self.custom_weights = {}
            
        # valid metrics check
        invalid_metrics = set(self.selected_metrics) - set(self.available_metrics.keys())
        if invalid_metrics:
            raise ValueError(f"Invalid metrics selected: {invalid_metrics}")
    
    def get_weights(self) -> List[float]:
        weights = []
        for metric in self.selected_metrics:
            if metric in self.custom_weights:
                weights.append(self.custom_weights[metric])
            else:
                weights.append(self.available_metrics[metric][0])
        return weights
    
    def is_minimize_metric(self, metric: str) -> bool:
        return self.available_metrics[metric][1]
    
    def get_num_objectives(self) -> int:
        return len(self.selected_metrics)

def calc_metrics(pfs, benchmark, params):
    returns = pfs.returns()
    sortino = returns.vbt.returns.sortino_ratio()
    sharpe = returns.vbt.returns.sharpe_ratio()
    rel_drawdown = pfs.max_drawdown() / benchmark.max_drawdown()
    drawdown = pfs.max_drawdown()
    alpha = pfs.annualized_return() - benchmark.annualized_return()
    annual_return = pfs.annualized_return()
    position_stability = [np.sum(p.position_sizing ** 2) for p in params]
    position_stability = Series(position_stability, index=sortino.index)

    return sortino, sharpe, rel_drawdown, drawdown, alpha, annual_return, position_stability

def fitness(sortino, sharpe, rel_drawdown, alpha, position_stability):
    # removes non-pareto front individuals in the bottom 10% performers in any metric

    rel_drawdown = rel_drawdown.clip(lower=0.4) 
    
    sort_arr = sortino.to_numpy()
    shar_arr = sharpe.to_numpy()
    dd_arr   = rel_drawdown.to_numpy()
    alpha_arr= alpha.to_numpy()
    pos_stab_arr = position_stability.to_numpy()

    valid_mask = (
        np.isfinite(sortino) & 
        np.isfinite(sharpe) & 
        np.isfinite(rel_drawdown) &
        np.isfinite(alpha) & 
        np.isfinite(position_stability)
    )
        
    sortino_10th = np.percentile(sortino[valid_mask], 10)
    sharpe_10th = np.percentile(sharpe[valid_mask], 10)
    alpha_10th = np.percentile(alpha[valid_mask], 10)
    rel_dd_90th = np.percentile(rel_drawdown[valid_mask], 90)  # 90th percentile for drawdown (higher is worse)
    pos_stab_90th = np.percentile(position_stability[valid_mask], 90)  # 90th percentile for position stability (higher is worse)
    
    bottom_10_mask = (
        (sortino < sortino_10th) |
        (sharpe < sharpe_10th) |
        (rel_drawdown > rel_dd_90th) |
        (alpha < alpha_10th) |
        (position_stability > pos_stab_90th)
    )
        
    def find_pareto_mask(sort_arr, shar_arr, dd_arr, alpha_arr, pos_stab_arr):
        sort_comp = sort_arr[:, np.newaxis] >= sort_arr[np.newaxis, :]
        shar_comp = shar_arr[:, np.newaxis] >= shar_arr[np.newaxis, :]
        alpha_comp = alpha_arr[:, np.newaxis] >= alpha_arr[np.newaxis, :]
        dd_comp = dd_arr[:, np.newaxis] <= dd_arr[np.newaxis, :]
        pos_comp = pos_stab_arr[:, np.newaxis] <= pos_stab_arr[np.newaxis, :]
        
        # Check if i dominates j for all pairs
        better_or_eq = sort_comp & shar_comp & alpha_comp & dd_comp & pos_comp
        
        # Check strict dominance
        strictly_better = (
            (sort_arr[:, np.newaxis] > sort_arr[np.newaxis, :]) |
            (shar_arr[:, np.newaxis] > shar_arr[np.newaxis, :]) |
            (alpha_arr[:, np.newaxis] > alpha_arr[np.newaxis, :]) |
            (dd_arr[:, np.newaxis] < dd_arr[np.newaxis, :]) |
            (pos_stab_arr[:, np.newaxis] < pos_stab_arr[np.newaxis, :])
        )
        
        dominates = better_or_eq & strictly_better
        
        pareto_mask = ~np.any(dominates, axis=0)
        return pareto_mask
    
    pareto_mask = find_pareto_mask(sort_arr, shar_arr, dd_arr, alpha_arr, pos_stab_arr)
            
    # Combine validity and bottom 10% filtering
    final_mask = valid_mask & (~bottom_10_mask | pareto_mask)
    
    fitness_sortino = np.where(final_mask, sortino, -1000)
    fitness_sharpe = np.where(final_mask, sharpe, -1000)
    fitness_rel_drawdown = np.where(final_mask, rel_drawdown, 1000)
    fitness_alpha = np.where(final_mask, alpha, -1000)
    fitness_position_stability = np.where(final_mask, position_stability, 1000)

    return (fitness_sortino, fitness_sharpe, fitness_rel_drawdown, fitness_alpha, fitness_position_stability)