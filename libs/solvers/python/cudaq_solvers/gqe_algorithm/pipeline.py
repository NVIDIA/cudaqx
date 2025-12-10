# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import torch
import cudaq
import lightning as L
from mpi4py import MPI
from torch.nn import functional as F
from lightning import LightningModule
from .data import ReplayBuffer, BufferDataset
from torch.utils.data import DataLoader
from torch.distributions import Categorical


class Pipeline(LightningModule):
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

    def __init__(self, cfg, cost, pool, model, factory, numQPUs=1):
        super().__init__()

        # Set seed for reproducibility
        L.seed_everything(cfg.seed)

        self.numQPUs = numQPUs
        self.cfg = cfg
        self.model = model.to(self.device)
        self.factory = factory
        self.pool = pool
        self.benchmark_energy = cfg.benchmark_energy
        self._cost = cost
        self.loss = self.factory.create_loss_fn(cfg).to(self.device)
        self.scheduler = self.factory.create_temperature_scheduler(self.cfg)
        self.ngates = cfg.ngates
        self.num_samples = cfg.num_samples
        self.buffer = ReplayBuffer(size=cfg.buffer_size)
        self.save_hyperparameters(ignore=['cost', 'pool', 'model', 'factory'])
        self._starting_idx = torch.zeros(self.num_samples,
                                         1,
                                         dtype=torch.long,
                                         device=self.device)

    def configure_optimizers(self):
        """Configure optimizer for training.

        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)

    def on_fit_start(self):
        # Recreate _starting_idx with correct device after model setup
        self._starting_idx = torch.zeros(self.num_samples,
                                         1,
                                         dtype=torch.long,
                                         device=self.device)
        while len(self.buffer) < self.cfg.warmup_size:
            self.collect_rollout()
        super().on_fit_start()

    def on_train_epoch_start(self):
        self.collect_rollout()

    def collect_rollout(self):
        idx_output = self.generate()
        energies = self.computeCost(idx_output[:, 1:], self.pool)
        for seq, energy in zip(idx_output, energies):
            self.buffer.push(seq, energy)
        self.scheduler.update(energies=energies)

    def set_cost(self, cost):
        """Set the cost function used to evaluate operator sequences.

        Args:
            cost: New cost function to use
        """
        self._cost = cost

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

    def training_step(self, batch, batch_idx):
        """Perform one training step.

        Lightning calls this method during training with batches from train_dataloader.

        Args:
            batch: Dictionary containing 'idx' (sequences) and 'energy' (energy values)
            batch_idx: Index of current batch

        Returns:
            torch.Tensor: Loss value for this batch
        """
        # Move batch data to device
        idx = batch["idx"].to(self.device)
        energies = batch["energy"].to(self.device)

        log_values = {}
        logits = self.model(idx).logits
        loss = self.loss.compute(
            energies,
            logits,
            idx[:, 1:],
            log_values,
            inverse_temperature=self.scheduler.get_inverse_temperature(),
            current_step=batch_idx)

        # Log metrics
        self.log_dict(log_values, prog_bar=False, on_step=False, on_epoch=True)
        self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("energy_mean",
                 energies.mean(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False)
        self.log("energy_min",
                 energies.min(),
                 prog_bar=False,
                 on_epoch=True,
                 on_step=False)
        self.log("inverse_temperature",
                 self.scheduler.get_inverse_temperature(),
                 prog_bar=True,
                 on_epoch=True,
                 on_step=False)

        return loss

    def train_dataloader(self):
        return DataLoader(
            BufferDataset(self.buffer, self.cfg.step_per_epoch),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=
            0,  # Avoid multiprocessing to prevent pickling issues with CUDA-Q objects
        )

    def generate(self, idx=None, ngates=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        if idx is None:
            idx = self._starting_idx.clone()
        if ngates is None:
            ngates = self.ngates
        current_temp = self.scheduler.get_inverse_temperature()
        for _ in range(ngates):
            idx_cond = idx
            logits_base = self.model(idx_cond)
            logits = logits_base.logits[:, -1, :]
            probs = Categorical(logits=-current_temp * logits)
            idx_next = probs.sample()
            idx = torch.cat((idx, idx_next.unsqueeze(1)), dim=1)
        return idx

    def logits(self, idx):
        logits_base = self.model(idx)
        idx = idx[:, 1:]
        return torch.gather(logits_base.logits, 2,
                            idx.unsqueeze(-1)).squeeze(-1)
