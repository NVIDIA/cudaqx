from .loss import ExpLogitMatching, GFlowLogitMatching, GRPOLoss
from .scheduler import DefaultScheduler, CosineScheduler, VarBasedScheduler

class Factory:
    def create_loss_fn(self, cfg):
        if cfg.loss == "exp":
            return ExpLogitMatching(cfg.energy_offset)
        elif cfg.loss == "grpo":
            clip_ratio = getattr(cfg, 'clip_ratio', 0.2)
            return GRPOLoss(clip_ratio=clip_ratio)
        elif cfg.loss == "gflow":
            return GFlowLogitMatching(cfg.energy_offset)
        else:
            raise ValueError(f"Invalid loss function: {cfg.loss}")
    
    def create_temperature_scheduler(self, cfg):
        """Create temperature scheduler based on configuration.
        
        Args:
            cfg: Configuration object with temperature parameters
            
        Returns:
            TemperatureScheduler: Scheduler instance
        """
        scheduler_type = getattr(cfg, 'scheduler', 'default')
        
        if scheduler_type == 'cosine':
            minimum = getattr(cfg, 'temperature_min', cfg.temperature)
            maximum = getattr(cfg, 'temperature_max', cfg.temperature + 1.0)
            frequency = getattr(cfg, 'scheduler_frequency', 100)
            return CosineScheduler(minimum, maximum, frequency)
        elif scheduler_type == 'variance':
            target_var = getattr(cfg, 'target_variance', 0.1)
            return VarBasedScheduler(cfg.temperature, cfg.del_temperature, target_var)
        elif scheduler_type == 'default':
            return DefaultScheduler(cfg.temperature, cfg.del_temperature)
        else:
            raise ValueError(f"Invalid scheduler type: {scheduler_type}")