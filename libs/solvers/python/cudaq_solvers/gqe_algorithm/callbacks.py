# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys
import torch
from lightning.pytorch.callbacks import Callback


class MinEnergyCallback(Callback):
    """Callback to track minimum energy found during training.
    
    Keeps track of the minimum energy value and corresponding operator indices
    across all training epochs.
    """

    def __init__(self):
        super().__init__()
        self.min_energy = sys.maxsize
        self.min_indices = None
        self.min_energy_history = []

    def on_train_epoch_end(self, trainer, pl_module):
        """Update minimum energy after each epoch.
        
        Args:
            trainer: Lightning trainer instance
            pl_module: The Pipeline module being trained
        """
        # Get energies from the buffer
        if len(pl_module.buffer) > 0:
            # Check recent energies added to buffer
            for i in range(max(0, len(pl_module.buffer) - pl_module.num_samples), len(pl_module.buffer)):
                seq, energy = pl_module.buffer.buf[i]
                if isinstance(energy, torch.Tensor):
                    energy = energy.item()
                if energy < self.min_energy:
                    self.min_energy = energy
                    self.min_indices = seq
            
            self.min_energy_history.append(self.min_energy)
            
            # Log to trainer if logging is enabled
            if trainer.logger is not None:
                trainer.logger.log_metrics({
                    "min_energy": self.min_energy,
                    "temperature": pl_module.scheduler.get_inverse_temperature()
                }, step=trainer.current_epoch)

    def get_results(self):
        """Get the minimum energy and corresponding indices.
        
        Returns:
            tuple: (min_energy, min_indices)
        """
        return self.min_energy, self.min_indices


class TrajectoryCallback(Callback):
    """Callback to save training trajectory data.
    
    Records loss, energies, and indices for each training step and saves
    to a file at the end of training.
    
    Args:
        trajectory_file_path: Path to save trajectory data
    """

    def __init__(self, trajectory_file_path):
        super().__init__()
        self.trajectory_file_path = trajectory_file_path
        self.trajectory_data = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Record trajectory data after each training batch.
        
        Args:
            trainer: Lightning trainer instance
            pl_module: The Pipeline module being trained
            outputs: Training step outputs
            batch: Current batch data
            batch_idx: Index of current batch
        """
        # Record the batch data
        if outputs is not None and 'loss' in outputs:
            loss = outputs['loss']
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            
            # Get indices and energies from batch
            indices = batch.get('idx', None)
            energies = batch.get('energy', None)
            
            if indices is not None and energies is not None:
                if isinstance(indices, torch.Tensor):
                    indices = indices.cpu().numpy().tolist()
                if isinstance(energies, torch.Tensor):
                    energies = energies.cpu().numpy().tolist()
                
                self.trajectory_data.append({
                    'epoch': trainer.current_epoch,
                    'batch_idx': batch_idx,
                    'loss': loss,
                    'indices': indices,
                    'energies': energies
                })

    def on_train_end(self, trainer, pl_module):
        """Save trajectory data to file at end of training.
        
        Args:
            trainer: Lightning trainer instance
            pl_module: The Pipeline module being trained
        """
        import json
        import os
        
        os.makedirs(os.path.dirname(self.trajectory_file_path), exist_ok=True)
        if os.path.exists(self.trajectory_file_path):
            print(f"Warning: Overwriting existing trajectory file at {self.trajectory_file_path}")
        
        with open(self.trajectory_file_path, 'w') as f:
            for data in self.trajectory_data:
                f.write(json.dumps(data) + '\n')
        
        print(f"Trajectory data saved to {self.trajectory_file_path}")


