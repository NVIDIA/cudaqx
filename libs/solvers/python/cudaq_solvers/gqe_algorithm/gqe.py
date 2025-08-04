# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from .transformer import Transformer
import torch
import lightning as L
from abc import ABC, abstractmethod
import math, json, sys, time, os
from ml_collections import ConfigDict
import cudaq

torch.set_float32_matmul_precision('high')


class TemperatureScheduler(ABC):
    """Abstract base class for temperature scheduling in GQE.
    
    Temperature scheduling controls how the temperature parameter changes during training,
    which affects the exploration vs exploitation trade-off in operator selection.
    """

    @abstractmethod
    def get(self, iter):
        """Get temperature value for the given iteration.
        
        Args:
            iter: Current iteration number
            
        Returns:
            float: Temperature value for this iteration
        """
        pass


class DefaultScheduler(TemperatureScheduler):
    """Linear temperature scheduler that increases temperature by a fixed delta each iteration.
    
    Args:
        start: Initial temperature value
        delta: Amount to increase temperature each iteration
    """

    def __init__(self, start, delta) -> None:
        self.start = start
        self.delta = delta

    def get(self, iter):
        """Get linearly increasing temperature value.
        
        Args:
            iter: Current iteration number
            
        Returns:
            float: start + iter * delta
        """
        return self.start + iter * self.delta


class CosineScheduler(TemperatureScheduler):
    """Cosine-based temperature scheduler that oscillates between min and max values.
    
    Useful for periodic exploration and exploitation phases during training.
    
    Args:
        minimum: Minimum temperature value
        maximum: Maximum temperature value
        frequency: Number of iterations for one complete cycle
    """

    def __init__(self, minimum, maximum, frequency) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.frequency = frequency

    def get(self, iter):
        """Get temperature value following a cosine curve.
        
        Args:
            iter: Current iteration number
            
        Returns:
            float: Temperature value between minimum and maximum following cosine curve
        """
        return (self.maximum + self.minimum) / 2 - (
            self.maximum - self.minimum) / 2 * math.cos(
                2 * math.pi * iter / self.frequency)


class TrajectoryData:
    """Container for training trajectory data at a single iteration.
    
    Stores loss value, selected operator indices, and corresponding energies
    for a single training iteration.
    
    Args:
        iter_num: Iteration number
        loss: Loss value for this iteration
        indices: Selected operator indices
        energies: Corresponding energy values
    """

    def __init__(self, iter_num, loss, indices, energies):
        self.iter_num = iter_num
        self.loss = loss
        self.indices = indices
        self.energies = energies

    def to_json(self):
        map = {
            "iter": self.iter_num,
            "loss": self.loss,
            "indices": self.indices,
            "energies": self.energies
        }
        return json.dumps(map)

    @classmethod
    def from_json(self, string):
        if string.startswith('"'):
            string = string[1:len(string) - 1]
            string = string.replace("\\", "")
        map = json.loads(string)
        return TrajectoryData(map["iter"], map["loss"], map["indices"],
                              map["energies"])


class FileMonitor:
    """Records and saves training trajectory data.
    
    Maintains a list of TrajectoryData objects and can save them to a file,
    allowing training progress to be analyzed or training to be resumed.
    """

    def __init__(self):
        self.lines = []

    def record(self, iter_num, loss, energies, indices):
        """Record trajectory data for one iteration.
        
        Args:
            iter_num: Current iteration number
            loss: Loss value for this iteration
            energies: List of energy values
            indices: List of selected operator indices
        """
        energies = energies.cpu().numpy().tolist()
        indices = indices.cpu().numpy().tolist()
        data = TrajectoryData(iter_num, loss.item(), indices, energies)
        self.lines.append(data.to_json())

    def save(self, path):
        """Save all recorded trajectory data to a file.
        
        Args:
            path: Path to save the trajectory data file
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            print(f"Warning: Overwriting existing trajectory file at {path}")
        with open(path, 'w') as f:
            for l in self.lines:
                f.write(f"{l}\n")


def validate_config(cfg: ConfigDict):
    """Validate all configuration parameters for GQE.
    
    Checks that all required parameters exist and have valid values.
    
    Args:
        cfg: Configuration object to validate
        
    Raises:
        ValueError: If any configuration parameter is invalid
    """
    # Basic parameters
    if cfg.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if cfg.max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if cfg.ngates <= 0:
        raise ValueError("ngates must be positive")

    # Learning parameters
    if cfg.lr <= 0:
        raise ValueError("learning rate must be positive")
    if cfg.grad_norm_clip <= 0:
        raise ValueError("grad_norm_clip must be positive")

    # Temperature parameters
    if cfg.temperature <= 0:
        raise ValueError("temperature must be positive")
    if cfg.del_temperature == 0:
        raise ValueError("del_temperature cannot be zero")

    # Dropout parameters (must be probabilities)
    if not (0 <= cfg.resid_pdrop <= 1):
        raise ValueError("resid_pdrop must be between 0 and 1")
    if not (0 <= cfg.embd_pdrop <= 1):
        raise ValueError("embd_pdrop must be between 0 and 1")
    if not (0 <= cfg.attn_pdrop <= 1):
        raise ValueError("attn_pdrop must be between 0 and 1")


def get_default_config():
    """Create a default configuration for GQE.
    
    Args:
        num_samples (int): Number of circuits to generate during each epoch/batch. Default=5
        max_iters (int): Number of epochs to run. Default=100
        ngates (int): Number of gates that make up each generated circuit. Default=20
        seed (int): Random seed. Default=3047
        lr (float): Learning rate used by the optimizer. Default=5e-7
        energy_offset (float): Offset added to expectation value of the circuit (Energy) for numerical 
            stability, see `K. Nakaji et al. (2024) <https://arxiv.org/abs/2401.09253>`_ Sec. 3. Default=0.0
        grad_norm_clip (float): max_norm for clipping gradients, see `Lightning docs <https://lightning.ai/docs/fabric/stable/api/fabric_methods.html#clip-gradients>`_. Default=1.0
        temperature (float): Starting inverse temperature β as described in `K. Nakaji et al. (2024) <https://arxiv.org/abs/2401.09253>`_ 
            Sec. 2.2. Default=5.0
        del_temperature (float): Temperature increase after each epoch. Default=0.05
        resid_pdrop (float): The dropout probability for all fully connected layers in the embeddings, 
            encoder, and pooler, see `GPT2Config <https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py>`_. Default=0.0
        embd_pdrop (float): The dropout ratio for the embeddings, see `GPT2Config <https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py>`_. Default=0.0
        attn_pdrop (float): The dropout ratio for the attention, see `GPT2Config <https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/configuration_gpt2.py>`_. Default=0.0
        small (bool): Uses a small transformer (6 hidden layers and 6 attention heads as opposed to 
            the default transformer of 12 of each). Default=False
        use_fabric_logging (bool): Whether to enable fabric logging. Default=False
        fabric_logger (object): Fabric logger to use for logging. If None, no logging will be done. Default=None
        save_trajectory (bool): Whether to save the trajectory data to a file. Default=False
        trajectory_file_path (str): Path to save the trajectory data file. Default="gqe_logs/gqe_trajectory.json"
        verbose (bool): Enable verbose output to the console. Output includes the epoch, loss, 
            model.train_step time, and minimum energy. Default=False
        
    Returns:
        ConfigDict: Default configuration for GQE
    """
    cfg = ConfigDict()
    cfg.num_samples = 5  # akin to batch size
    cfg.max_iters = 100
    cfg.ngates = 20
    cfg.seed = 3047
    cfg.lr = 5e-7
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 5.0
    cfg.del_temperature = 0.05
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = False
    cfg.use_fabric_logging = False  # Whether to enable fabric logging
    cfg.fabric_logger = None  # Fabric logger
    cfg.save_trajectory = False  # Whether to save trajectory data
    cfg.trajectory_file_path = "gqe_logs/gqe_trajectory.json"  # Path to save trajectory data
    cfg.verbose = False
    return cfg


def __internal_run_gqe(temperature_scheduler: TemperatureScheduler,
                       cfg: ConfigDict, model, pool, optimizer):
    """Internal implementation of the GQE training loop.
    
    Args:
        temperature_scheduler: Optional scheduler for temperature parameter
        cfg: Configuration object
        model: The transformer model to train
        pool: Pool of quantum operators to select from
        optimizer: Optimizer for model parameters
        
    Returns:
        tuple: (minimum energy found, corresponding operator indices)
    """
    # Configure Fabric with optional logging
    fabric_kwargs = {"accelerator": "auto", "devices": 1}
    if cfg.use_fabric_logging:
        if cfg.fabric_logger is None:
            raise ValueError(
                "Fabric Logger is not set. Please set it in the config by providing a logger to `cfg.fabric_logger`."
            )
        fabric_kwargs["loggers"] = [cfg.fabric_logger]
    else:
        fabric_kwargs["loggers"] = False

    fabric = L.Fabric(**fabric_kwargs)
    fabric.seed_everything(cfg.seed)
    fabric.launch()
    if cfg.save_trajectory:
        monitor = FileMonitor()
    else:
        monitor = None
    model, optimizer = fabric.setup(model, optimizer)
    model.mark_forward_method('train_step')
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    if cfg.verbose:
        print(f"total trainable params: {pytorch_total_params / 1e6:.2f}M")
    min_energy = sys.maxsize
    min_indices = None
    for epoch in range(cfg.max_iters):
        optimizer.zero_grad()
        start = time.time()
        loss, energies, indices, log_values = model.train_step(pool)
        if cfg.verbose:
            print('epoch', epoch, 'loss', loss, 'model.train_step time:',
                  time.time() - start, torch.min(energies))
        if monitor is not None:
            monitor.record(epoch, loss, energies, indices)
        for e, indices in zip(energies, indices):
            energy = e.item()
            if energy < min_energy:
                min_energy = e.item()
                min_indices = indices
        if cfg.use_fabric_logging:
            log_values[f"min_energy at"] = min_energy
            log_values[f"temperature at"] = model.temperature
            log_values[f"loss at"] = loss
            fabric.log_dict(log_values, step=epoch)
        fabric.backward(loss)
        fabric.clip_gradients(model, optimizer, max_norm=cfg.grad_norm_clip)
        optimizer.step()
        if temperature_scheduler is not None:
            model.temperature = temperature_scheduler.get(epoch)
        else:
            model.temperature += cfg.del_temperature
    model.set_cost(None)
    min_indices = min_indices.cpu().numpy().tolist()
    if cfg.use_fabric_logging:
        fabric.log('circuit', json.dumps(min_indices))
    if cfg.save_trajectory:
        monitor.save(cfg.trajectory_file_path)
    return min_energy, min_indices


def gqe(cost, pool, config=None, **kwargs):
    """Run the Generative Quantum Eigensolver algorithm.
    
    GQE uses a transformer model to learn which quantum operators from a pool
    should be applied to minimize a given cost function. Python-only implementation.

    The GQE implementation in CUDA-Q Solvers is based on this paper: `K. Nakaji et al. (2024) <https://arxiv.org/abs/2401.09253>`_.
    
    Args:
        cost: Cost function that evaluates operator sequences
        pool: List of quantum operators to select from
        config: Optional configuration object. If None, uses kwargs to override defaults
        **kwargs: Optional keyword arguments to override default configuration. The following
            special arguments are supported:
            
            - model: Can pass in an already constructed transformer
            - optimizer: Can pass in an already constructed optimizer
            
            Additionally, any default config parameter can be overridden via kwargs if no
            config object is provided, for example:
            
            - max_iters: Overrides cfg.max_iters for total number of epochs to run
            - energy_offset: Overrides cfg.energy_offset for offset to add to expectation value
        
    Returns:
        tuple: Minimum energy found, corresponding operator indices
    """
    cfg = get_default_config()

    if config == None:
        [
            setattr(cfg, a, kwargs[a])
            for a in dir(cfg)
            if not a.startswith('_') and a in kwargs
        ]
    else:
        cfg = config

    validate_config(cfg)

    # Don't let someone override the vocab_size
    cfg.vocab_size = len(pool)
    cudaqTarget = cudaq.get_target()
    numQPUs = cudaqTarget.num_qpus()
    model = Transformer(
        cfg, cost, loss='exp',
        numQPUs=numQPUs) if 'model' not in kwargs else kwargs['model']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr) if 'optimizer' not in kwargs else kwargs['optimizer']
    return __internal_run_gqe(None, cfg, model, pool, optimizer)
