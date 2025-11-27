# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from .pipeline import Pipeline
from .model import GPT2
from .factory import Factory
import torch
import lightning as L
import json
import os
from ml_collections import ConfigDict
import cudaq

torch.set_float32_matmul_precision('high')


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
            for line in self.lines:
                f.write(f"{line}\n")


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
        use_lightning_logging (bool): Whether to enable lightning logging. Default=False
        fabric_logger (object): Fabric logger to use for logging. If None, no logging will be done. Default=None
        save_trajectory (bool): Whether to save the trajectory data to a file. Default=False
        trajectory_file_path (str): Path to save the trajectory data file. Default="gqe_logs/gqe_trajectory.json"
        verbose (bool): Enable verbose output to the console. Output includes the epoch, loss,
            model.train_step time, and minimum energy. Default=False
        buffer_size (int): Size of replay buffer for storing trajectories. Default=50
        warmup_size (int): Initial buffer warmup size before training starts. Default=50
        trainer.step_per_epoch (int): Number of training steps per epoch. Default=10
        trainer.batch_size (int): Batch size for training. Default=50
        
    Returns:
        ConfigDict: Default configuration for GQE
    """
    cfg = ConfigDict()
    cfg.num_samples = 20  # akin to batch size
    cfg.max_iters = 100
    cfg.ngates = 20
    cfg.seed = 3047
    cfg.lr = 5e-7
    cfg.energy_offset = 0.0
    cfg.grad_norm_clip = 1.0
    cfg.temperature = 0.5
    cfg.del_temperature = 0.02
    cfg.resid_pdrop = 0.0
    cfg.embd_pdrop = 0.0
    cfg.attn_pdrop = 0.0
    cfg.small = False
    cfg.use_lightning_logging = False  # Whether to enable lightning logging
    cfg.lightning_logger = None  # Lightning logger
    cfg.save_trajectory = False  # Whether to save trajectory data
    cfg.trajectory_file_path = "gqe_logs/gqe_trajectory.json"  # Path to save trajectory data
    cfg.verbose = False
    cfg.loss = "grpo"
    # Replay buffer parameters
    cfg.buffer_size = 20  # Size of replay buffer
    cfg.warmup_size = 20  # Initial buffer warmup size
    # Trainer parameters
    cfg.trainer = ConfigDict()
    cfg.trainer.step_per_epoch = 20  # Steps per training epoch
    cfg.trainer.batch_size = 20  # Batch size for training
    return cfg


def __internal_run_gqe(cfg: ConfigDict, pipeline, pool):
    """Internal implementation of the GQE training loop using Lightning Trainer.
    
    Args:
        cfg: Configuration object
        pipeline: The Pipeline module containing model and training logic
        pool: Pool of quantum operators to select from
        
    Returns:
        tuple: (minimum energy found, corresponding operator indices)
    """
    from .callbacks import MinEnergyCallback, TrajectoryCallback
    
    # Configure trainer kwargs
    trainer_kwargs = {
        "accelerator": "auto",
        "devices": 1,
        "max_epochs": cfg.max_iters,
        "gradient_clip_val": cfg.grad_norm_clip,
        "enable_progress_bar": cfg.verbose,
        "enable_model_summary": cfg.verbose,
        "enable_checkpointing": False,  # Disable checkpointing for speed
        "log_every_n_steps": 1,
        "num_sanity_val_steps": 0,  # Disable validation sanity checks
    }
    
    # Set up logging
    if cfg.use_lightning_logging:
        if cfg.lightning_logger is None:
            raise ValueError(
                "Lightning Logger is not set. Please set it in the config by providing a logger to `cfg.lightning_logger`."
            )
        trainer_kwargs["logger"] = cfg.lightning_logger
    else:
        trainer_kwargs["logger"] = False
    
    # Set up callbacks
    callbacks = []
    min_energy_callback = MinEnergyCallback()
    callbacks.append(min_energy_callback)
    
    if cfg.save_trajectory:
        trajectory_callback = TrajectoryCallback(cfg.trajectory_file_path)
        callbacks.append(trajectory_callback)
    
    trainer_kwargs["callbacks"] = callbacks
    
    # Create trainer
    trainer = L.Trainer(**trainer_kwargs)
    
    # Print model parameters if verbose
    if cfg.verbose:
        pytorch_total_params = sum(
            p.numel() for p in pipeline.model.parameters() if p.requires_grad)
        print(f"total trainable params: {pytorch_total_params / 1e6:.2f}M")
    
    # Train the model
    trainer.fit(pipeline)
    
    # Get results from callback
    min_energy, min_indices = min_energy_callback.get_results()
    
    # Convert indices to list if needed
    if min_indices is not None and isinstance(min_indices, torch.Tensor):
        min_indices = min_indices.cpu().numpy().tolist()
    
    # Log final circuit if logging is enabled
    if cfg.use_lightning_logging and min_indices is not None:
        trainer.logger.log_metrics({'circuit': json.dumps(min_indices)})
    
    # Clean up
    pipeline.set_cost(None)
    
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
            
            Additionally, any default config parameter can be overridden via kwargs if no
            config object is provided, for example:
            
            - max_iters: Overrides cfg.max_iters for total number of epochs to run
            - energy_offset: Overrides cfg.energy_offset for offset to add to expectation value
        
    Returns:
        tuple: Minimum energy found, corresponding operator indices
    """
    cfg = get_default_config()

    if config is None:
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
    factory = Factory()
    model = GPT2(cfg.small, cfg.vocab_size) if 'model' not in kwargs else kwargs['model']
    pipeline = Pipeline(cfg, cost, pool, model, factory, numQPUs=numQPUs)
    return __internal_run_gqe(cfg, pipeline, pool)
