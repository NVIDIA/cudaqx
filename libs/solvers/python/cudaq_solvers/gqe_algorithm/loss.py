# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from abc import abstractmethod, ABC
import torch


class LogitMatchingLoss(ABC):
    """Abstract base class for logit-energy matching loss functions.
    
    Loss functions in GQE compare the model's logits (predictions) with
    computed energies to guide the model toward selecting operator
    sequences that minimize energy.
    """

    @abstractmethod
    def compute(self, energies, logits_tensor, log_values):
        pass


class ExpLogitMatching(LogitMatchingLoss):
    """Simple exponential matching between logits and energies.
    
    Computes loss by comparing exponential of negative logits with
    exponential of negative energies from circuit evaluation. The energies
    represent the expectation values of the Hamiltonian problem operator
    obtained from quantum circuit execution during GPT training.
    
    Args:
        energy_offset: Offset added to expectation values of the circuit (Energy)
                      for numerical stability during training. 
        label: Label for logging purposes
    """

    def __init__(self, energy_offset, label) -> None:
        self._label = label
        self.energy_offset = energy_offset
        self.loss_fn = torch.nn.MSELoss()

    def compute(self, energies, logits_tensor, log_values):
        mean_logits = torch.mean(logits_tensor, 1)
        log_values[f"mean_logits at {self._label}"] = torch.mean(
            mean_logits - self.energy_offset)
        log_values[f"mean energy at {self._label}"] = torch.mean(energies)
        mean_logits = torch.mean(logits_tensor, 1)
        device = mean_logits.device
        return self.loss_fn(
            torch.exp(-mean_logits),
            torch.exp(-energies.to(device) - self.energy_offset))


class GFlowLogitMatching(LogitMatchingLoss):
    """Advanced logit-energy matching with learnable offset.
    
    Similar to ExpLogitMatching but learns an additional energy offset
    parameter during training, allowing for better adaptation to the
    energy scale.
    
    Args:
        energy_offset: Initial energy offset
        device: Device to place tensors on
        label: Label for logging purposes
        nn: Neural network module to register the offset parameter with
    """

    def __init__(self, energy_offset, device, label, nn: torch.nn) -> None:
        self._label = label
        self.loss_fn = torch.nn.MSELoss()
        self.energy_offset = energy_offset
        self.normalization = 10**-5
        self.param = torch.nn.Parameter(torch.tensor([0.0]).to(device))
        nn.register_parameter(name="energy_offset", param=self.param)

    def compute(self, energies, logits_tensor, log_values):
        mean_logits = torch.mean(logits_tensor, 1)
        energy_offset = self.energy_offset + self.param / self.normalization
        log_values[f"energy_offset at {self._label}"] = energy_offset
        log_values[f"mean_logits at {self._label}"] = torch.mean(mean_logits -
                                                                 energy_offset)
        log_values[f"mean energy at {self._label}"] = torch.mean(energies)
        mean_logits = torch.mean(logits_tensor, 1)
        device = mean_logits.device
        loss = self.loss_fn(
            torch.exp(-mean_logits),
            torch.exp(-(energies.to(device) + energy_offset.to(device))))
        return loss