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
    """Calculate all available metrics"""
    returns = pfs.returns()
    sortino = returns.vbt.returns.sortino_ratio()
    sharpe = returns.vbt.returns.sharpe_ratio()
    rel_drawdown = pfs.max_drawdown() / benchmark.max_drawdown()
    drawdown = pfs.max_drawdown()
    alpha = pfs.annualized_return() - benchmark.annualized_return()
    annual_return = pfs.annualized_return()
    position_stability = [np.sum(p.position_sizing ** 2) for p in params]
    position_stability = Series(position_stability, index=sortino.index)

    # Return dictionary of all metrics
    metrics = {
        'sortino': sortino,
        'sharpe': sharpe,
        'rel_drawdown': rel_drawdown,
        'drawdown': drawdown,
        'alpha': alpha,
        'annual_return': annual_return,
        'position_stability': position_stability
    }
    
    return metrics

def fitness(metrics_dict, config: FitnessConfig):
    """
    Flexible fitness function that works with configurable metrics
    """
    # Apply rel_drawdown clipping if it's selected
    if 'rel_drawdown' in metrics_dict:
        metrics_dict['rel_drawdown'] = metrics_dict['rel_drawdown'].clip(lower=0.4)
    
    # Extract selected metrics
    selected_data = {}
    for metric in config.selected_metrics:
        if metric not in metrics_dict:
            raise ValueError(f"Metric '{metric}' not found in calculated metrics")
        selected_data[metric] = metrics_dict[metric]
    
    # Convert to numpy arrays
    metric_arrays = {}
    for metric, data in selected_data.items():
        metric_arrays[metric] = data.to_numpy()
    
    # Create validity mask
    valid_mask = np.ones(len(next(iter(metric_arrays.values()))), dtype=bool)
    for arr in metric_arrays.values():
        valid_mask &= np.isfinite(arr)
    
    # Apply bottom percentile filtering if enabled
    if config.enable_bottom_percentile_filter:
        bottom_mask = np.zeros_like(valid_mask)
        
        for metric in config.selected_metrics:
            arr = metric_arrays[metric]
            is_minimize = config.is_minimize_metric(metric)
            
            if is_minimize:
                # For minimize metrics, filter out top percentile (worst performers)
                threshold = np.percentile(arr[valid_mask], config.top_percentile)
                bottom_mask |= (arr > threshold)
            else:
                # For maximize metrics, filter out bottom percentile (worst performers)  
                threshold = np.percentile(arr[valid_mask], config.bottom_percentile)
                bottom_mask |= (arr < threshold)
    else:
        bottom_mask = np.zeros_like(valid_mask)
    
    # Find Pareto front
    def find_pareto_mask(*arrays):
        n = len(arrays[0])
        pareto_mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            if not pareto_mask[i]:
                continue
                
            for j in range(n):
                if i == j or not pareto_mask[j]:
                    continue
                    
                # Check if j dominates i
                dominates = True
                strictly_better = False
                
                for k, (arr, metric) in enumerate(zip(arrays, config.selected_metrics)):
                    is_minimize = config.is_minimize_metric(metric)
                    
                    if is_minimize:
                        if arr[j] > arr[i]:  # j is worse than i
                            dominates = False
                            break
                        elif arr[j] < arr[i]:  # j is better than i
                            strictly_better = True
                    else:
                        if arr[j] < arr[i]:  # j is worse than i
                            dominates = False
                            break
                        elif arr[j] > arr[i]:  # j is better than i
                            strictly_better = True
                
                if dominates and strictly_better:
                    pareto_mask[i] = False
                    break
        
        return pareto_mask
    
    pareto_mask = find_pareto_mask(*[metric_arrays[m] for m in config.selected_metrics])
    
    # Combine validity and filtering
    final_mask = valid_mask & (~bottom_mask | pareto_mask)
    
    # Create fitness values
    fitness_values = []
    for metric in config.selected_metrics:
        arr = metric_arrays[metric]
        is_minimize = config.is_minimize_metric(metric)
        
        if is_minimize:
            # For minimize metrics, use large positive value for invalid
            fitness_arr = np.where(final_mask, arr, 1000)
        else:
            # For maximize metrics, use large negative value for invalid
            fitness_arr = np.where(final_mask, arr, -1000)
            
        fitness_values.append(fitness_arr)
    
    return tuple(fitness_values)