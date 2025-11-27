from .loss import ExpLogitMatching, GFlowLogitMatching
from .scheduler import DefaultScheduler, CosineScheduler

class Factory:
    def create_loss_fn(self, cfg, label):
        if cfg.loss == "exp":
            return ExpLogitMatching(cfg.energy_offset, label)
        else:
            return GFlowLogitMatching(cfg.energy_offset, label)
    
    def create_temperature_scheduler(self, cfg):
        """Create temperature scheduler based on configuration.
        
        Args:
            cfg: Configuration object with temperature parameters
            
        Returns:
            TemperatureScheduler: Scheduler instance
        """
        # Default to linear scheduler
        return DefaultScheduler(cfg.temperature, cfg.del_temperature)