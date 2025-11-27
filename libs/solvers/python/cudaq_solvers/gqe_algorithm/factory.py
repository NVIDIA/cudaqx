from .loss import ExpLogitMatching, GFlowLogitMatching, GRPOLoss
from .scheduler import DefaultScheduler, CosineScheduler

class Factory:
    def create_loss_fn(self, cfg):
        if cfg.loss == "exp":
            return ExpLogitMatching(cfg.energy_offset)
        elif cfg.loss == "grpo":
            clip_ratio = getattr(cfg, 'clip_ratio', 0.2)
            return GRPOLoss(clip_ratio=clip_ratio)
        else:
            return GFlowLogitMatching(cfg.energy_offset)
    
    def create_temperature_scheduler(self, cfg):
        """Create temperature scheduler based on configuration.
        
        Args:
            cfg: Configuration object with temperature parameters
            
        Returns:
            TemperatureScheduler: Scheduler instance
        """
        # Default to linear scheduler
        return DefaultScheduler(cfg.temperature, cfg.del_temperature)