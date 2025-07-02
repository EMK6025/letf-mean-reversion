# src/config.py
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class FitnessConfig:
    """Configuration for fitness function metrics and weights"""
    
    # Available metrics with their default weights and minimize flags
    # Format: metric_name: (weight, minimize)
    # weight: positive for maximize, negative for minimize in DEAP
    # minimize: True if lower values are better (used in fitness calculation)
    available_metrics = {
        'sortino': (1.0, False, -1.0),           # Higher is better
        'sharpe': (1.0, False, -1.0),            # Higher is better  
        'rel_drawdown': (-1.0, True, 2.0),      # Lower is better
        'alpha': (1.0, False, -.5),             # Higher is better
        'position_stability': (-1.0, True, 1.0), # Lower is better
        'drawdown': (-1.0, True, -1.0),          # Lower is better
        'annual_return': (1.0, False, -1.0)      # Higher is better
    }
    
    # Selected metrics for optimization (subset of available_metrics)
    selected_metrics: List[str] = None
    
    # Custom weights (optional override of defaults)
    custom_weights: Dict[str, float] = None
    
    # Filtering parameters
    enable_bottom_percentile_filter: bool = True
    bottom_percentile: float = 10.0
    top_percentile: float = 90.0  # For metrics where higher is worse
    
    def __post_init__(self):
        if self.selected_metrics is None:
            # Default selection - the original 5 metrics
            self.selected_metrics = ['sortino', 'sharpe', 'rel_drawdown', 'alpha', 'position_stability']
        
        if self.custom_weights is None:
            self.custom_weights = {}
            
        # Validate selected metrics
        invalid_metrics = set(self.selected_metrics) - set(self.available_metrics.keys())
        if invalid_metrics:
            raise ValueError(f"Invalid metrics selected: {invalid_metrics}")
    
    def get_weights(self) -> List[float]:
        """Get DEAP weights for selected metrics"""
        weights = []
        for metric in self.selected_metrics:
            if metric in self.custom_weights:
                weights.append(self.custom_weights[metric])
            else:
                weights.append(self.available_metrics[metric][0])
        return weights
    
    def is_minimize_metric(self, metric: str) -> bool:
        """Check if a metric should be minimized"""
        return self.available_metrics[metric][1]
    
    def get_num_objectives(self) -> int:
        """Get number of objectives for DEAP fitness creation"""
        return len(self.selected_metrics)