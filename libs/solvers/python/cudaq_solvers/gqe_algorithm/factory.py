from .loss import ExpLogitMatching, GFlowLogitMatching

class Factory:
    def create_loss_fn(self, cfg, label):
        if cfg.loss == "exp":
            return ExpLogitMatching(cfg.energy_offset, label)
        else:
            return GFlowLogitMatching(cfg.energy_offset, label)