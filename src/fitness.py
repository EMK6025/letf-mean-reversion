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


class AdaptiveFitness:
    def __init__(self, warmup_generations=5, target_generations=50):
        self.warmup_generations = warmup_generations
        self.target_generations = target_generations
        self.metrics_history = {
            'sortino': [],
            'sharpe': [],
            'rel_drawdown': [],
            'alpha': []
        }
    
    def __call__(self, sortino, sharpe, rel_drawdown, alpha, generation=0):
        return self.fitness(sortino, sharpe, rel_drawdown, alpha, generation)
    
    def fitness(self, sortino, sharpe, rel_drawdown, alpha, generation=0):
        self._update_metrics_history(sortino, sharpe, rel_drawdown, alpha)

        thresholds = self._get_adaptive_thresholds(generation)
        
        valid_mask = (
             np.isfinite(sortino) &
            np.isfinite(sharpe) &
            np.isfinite(alpha) &
            np.isfinite(rel_drawdown) &
            (sortino >= thresholds['sortino_min']) &
            (sharpe >= thresholds['sharpe_min']) &
            (alpha >= thresholds['alpha_min']) &
            (rel_drawdown <= thresholds['rel_drawdown_max'])
        )
        
        fitness_sortino = np.where(valid_mask, sortino, -1000)
        fitness_sharpe = np.where(valid_mask, sharpe, -1000)
        fitness_rel_drawdown = np.where(valid_mask, rel_drawdown, 1000)
        fitness_alpha = np.where(valid_mask, alpha, -1000)
        
        return (fitness_sortino, fitness_sharpe, fitness_rel_drawdown, fitness_alpha)
    
    def get_stats(self, generation):
        thresholds = self._get_adaptive_thresholds(generation)
        return {
            'generation': generation,
            'thresholds': thresholds,
            'history_length': len(self.metrics_history['sortino'])
        }
    
    def _update_metrics_history(self, sortino, sharpe, rel_drawdown, alpha):
        valid_mask = (sortino > -1000) & (sharpe > -1000) & (rel_drawdown < 1000) & (alpha > -1000)
        
        if np.any(valid_mask):
            self.metrics_history['sortino'].append(np.percentile(sortino[valid_mask], 90))
            self.metrics_history['sharpe'].append(np.percentile(sharpe[valid_mask], 90))
            self.metrics_history['rel_drawdown'].append(np.percentile(rel_drawdown[valid_mask], 10))  # Lower is better
            self.metrics_history['alpha'].append(np.percentile(alpha[valid_mask], 90))
    
    def _get_adaptive_thresholds(self, generation):
        thresholds = {
            'rel_drawdown_max': 1.0,
            'alpha_min': 0.0,
            'sortino_min': .3,
            'sharpe_min': .5
        }
        
        if len(self.metrics_history['sortino']) >= self.warmup_generations:
            # Use moving average of recent generations (up to 10)
            window = min(10, len(self.metrics_history['sortino']))
            
            recent_sortino = np.mean(self.metrics_history['sortino'][-window:])
            recent_sharpe = np.mean(self.metrics_history['sharpe'][-window:])
            recent_rel_dd = np.mean(self.metrics_history['rel_drawdown'][-window:])
            recent_alpha = np.mean(self.metrics_history['alpha'][-window:])
            
            # Progress from 20% to 80% of recent best performance
            progress_ratio = min(generation / self.target_generations, 1.0)
            threshold_pct = 0.2 + 0.6 * progress_ratio
            
            thresholds['sortino_min'] = recent_sortino * threshold_pct
            thresholds['sharpe_min'] = recent_sharpe * threshold_pct
            thresholds['alpha_min'] = recent_alpha * threshold_pct
            
            # For drawdown: start accepting 2x best, reduce to 1.2x
            dd_multiplier = 2.0 - 0.8 * progress_ratio
            thresholds['rel_drawdown_max'] = recent_rel_dd * dd_multiplier
        
        return thresholds


# backwards compatibility
fitness = AdaptiveFitness()

def create_fitness(warmup_generations=1, target_generations=50):
    """Create a custom fitness evaluator with different adaptation parameters"""
    return AdaptiveFitness(warmup_generations, target_generations)