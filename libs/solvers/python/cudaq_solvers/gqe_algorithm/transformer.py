# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import torch, cudaq
from mpi4py import MPI
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Config
from lightning import LightningModule
from .loss import ExpLogitMatching, GFlowLogitMatching

_device_cache = None


def get_device():
    """Determine the appropriate device for tensor operations.

    Probes the CUDA runtime to verify it is healthy.  If the context has
    been corrupted (e.g. by CUDA-Q initialisation) a single recovery is
    attempted by forcing PyTorch to re-initialise its CUDA state.  On
    failure a ``RuntimeError`` with actionable guidance is raised -- GQE
    requires a working GPU and silent CPU fallback would mask the problem
    with unacceptable performance.

    The result is cached so the probe only runs once per process.

    Returns:
        str: ``'cuda'``, ``'mps'``, or ``'cpu'``

    Raises:
        RuntimeError: If a CUDA GPU is detected but PyTorch cannot use it.
    """
    global _device_cache
    if _device_cache is not None:
        return _device_cache

    if torch.cuda.is_available():
        last_err = None
        for attempt in range(2):
            try:
                t = torch.tensor([1.0], device='cuda')
                _ = (t + t).item()
                del t
                torch.cuda.synchronize()
                _device_cache = 'cuda'
                return _device_cache
            except Exception as e:
                last_err = e
                if attempt == 0:
                    # Drain any pending async CUDA errors, then force
                    # PyTorch to re-establish the primary CUDA context.
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    if hasattr(torch.cuda, '_initialized'):
                        torch.cuda._initialized = False
                    try:
                        torch.cuda.init()
                    except Exception:
                        pass

        raise RuntimeError(
            f"CUDA GPU detected but PyTorch cannot use it: {last_err}\n\n"
            "This commonly happens when CUDA-Q (or another CUDA library) "
            "modifies the CUDA context before PyTorch is initialised.\n\n"
            "Workaround -- initialise PyTorch CUDA before importing cudaq:\n\n"
            "    import torch\n"
            "    torch.cuda.init()   # claim the primary CUDA context\n"
            "    import cudaq         # CUDA-Q inherits the existing context\n"
        ) from last_err

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _device_cache = 'mps'
    else:
        _device_cache = 'cpu'

    return _device_cache


class SmallConfig(GPT2Config):
    """Reduced-size configuration for GPT2 model.
    
    Uses fewer layers (6) and attention heads (6) than the default GPT2
    configuration, resulting in a smaller model that trains faster.
    
    Args:
        **kwargs: Additional GPT2 configuration parameters
    """

    def __init__(self, **kwargs):
        super().__init__(n_layer=6, n_head=6, **kwargs)


class Transformer(LightningModule):
    """GPT2-based transformer model for quantum operator selection.
    
    This model learns to select quantum operators from a pool to minimize
    a given cost function. It can be configured to use either a full-size
    or reduced-size architecture.
    
    Args:
        cfg: Configuration object containing model parameters
        cost: Cost function to evaluate operator sequences
        loss: Loss function type ('exp' or 'gflow')
        numQPUs: Number of QPUs available for cost evaluation
    """

    def __init__(self, cfg, cost, loss="exp", numQPUs=1):
        super().__init__()
        self._label = 'label_stand_in'
        self.numQPUs = numQPUs
        self.cfg = cfg
        gpt2cfg = GPT2Config(
            **{k: cfg[k] for k in GPT2Config().to_dict().keys() & cfg.keys()})
        if cfg.small:
            gpt2cfg = SmallConfig(
                **
                {k: cfg[k] for k in GPT2Config().to_dict().keys() & cfg.keys()})
        device = get_device()
        self.transformer = GPT2LMHeadModel(gpt2cfg).to(device)
        self.ngates = cfg.ngates
        self.num_samples = cfg.num_samples
        self.temperature = cfg.temperature
        self.save_hyperparameters()
        self._starting_idx = torch.zeros(self.num_samples,
                                         1,
                                         dtype=torch.int,
                                         device=device)
        if loss == "exp":
            self.loss = ExpLogitMatching(cfg.energy_offset, self._label)
        else:
            self.loss = GFlowLogitMatching(cfg.energy_offset, device,
                                           self._label, self)
        self._cost = cost

    def generate_logits(self, idx):
        """Generate logits for the next token given input indices.
        
        Args:
            idx: Input token indices
            
        Returns:
            torch.Tensor: Logits for next token prediction
        """
        logits = self.transformer(idx)[0]
        return logits

    def set_cost(self, cost):
        """Set the cost function used to evaluate operator sequences.
        
        Args:
            cost: New cost function to use
        """
        self._cost = cost

    def gather(self, idx, logits_base):
        """Gather logits for specific indices from base logits.
        
        Args:
            idx: Indices to gather logits for
            logits_base: Base logits to gather from
            
        Returns:
            torch.Tensor: Gathered logits
        """
        b_size = idx.shape[0]
        return torch.gather(logits_base, 2, idx.reshape(b_size, -1,
                                                        1)).reshape(b_size, -1)

    @torch.no_grad()
    def computeCost(self, idx_output, pool, **kwargs):
        """Compute cost for given operator sequences.
        
        Supports distributed computation using MPI if available.
        
        Args:
            idx_output: Indices of selected operators
            pool: Pool of quantum operators
            **kwargs: Additional arguments passed to cost function
            
        Returns:
            torch.Tensor: Computed costs for each sequence
            
        Raises:
            RuntimeError: If cost function returns invalid type
        """
        res = []
        if cudaq.mpi.is_initialized():
            rank = cudaq.mpi.rank()
            numRanks = cudaq.mpi.num_ranks()
            total_elements = len(idx_output)
            elements_per_rank = total_elements // numRanks
            remainder = total_elements % numRanks
            start = rank * elements_per_rank + min(rank, remainder)
            end = start + elements_per_rank + (1 if rank < remainder else 0)
            # This MPI rank owns rows[start:end]
            res = [
                self._cost([pool[j]
                            for j in row], qpu_id=i % self.numQPUs)
                for i, row in enumerate(idx_output[start:end])
            ]
        else:
            res = [
                self._cost([pool[j]
                            for j in row], qpu_id=i % self.numQPUs)
                for i, row in enumerate(idx_output)
            ]

        if isinstance(res[0], tuple) and len(res[0]) == 2:
            res = [
                getScalarFromHandleFunctor(handle)
                for (handle, getScalarFromHandleFunctor) in res
            ]

        if not isinstance(res[0], float):
            raise RuntimeError(
                'Invalid return type detected from user cost function.')

        # Need to perform MPI all gather here
        if cudaq.mpi.is_initialized():
            res = MPI.COMM_WORLD.allgather(res)
            res = [x for xs in res for x in xs]

        return torch.tensor(res, dtype=torch.float)

    def train_step(self,
                   pool,
                   indices=None,
                   energies=None,
                   numQPUs=None,
                   comm=None):
        """Perform one training step.
        
        Either generates new sequences and computes their costs,
        or uses provided sequences and energies for training.
        
        Args:
            pool: Pool of quantum operators
            indices: Optional pre-computed operator indices
            energies: Optional pre-computed energies
            numQPUs: Optional number of QPUs to use
            comm: Optional MPI communicator
            
        Returns:
            tuple: (loss, energies, indices, log_values)
        """
        log_values = {}
        if energies is not None:
            assert indices is not None
            idx_output = indices[:, 1:]
            logits_base = self.generate_logits(idx_output)
        else:
            idx_output, logits_base = self.generate()
            energies = self.computeCost(idx_output,
                                        pool,
                                        numQPUs=numQPUs,
                                        comm=comm)
        logits_tensor = self.gather(idx_output, logits_base)
        allLogits = logits_tensor

        loss = self.loss.compute(energies, allLogits, log_values)
        log_values[f"loss at {self._label}"] = loss
        return loss, energies, idx_output, log_values

    def generate(self, idx=None, ngates=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        if idx is None:
            idx = self._starting_idx.clone()
        condition_length = idx.size(dim=1)
        if ngates is None:
            ngates = self.ngates
        for _ in range(ngates):
            idx_cond = idx
            logits_base = self.generate_logits(idx_cond)
            logits = logits_base[:, -1, :]
            probs = F.softmax(-self.temperature * logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        idx = idx[:, condition_length:]
        return idx, logits_base
