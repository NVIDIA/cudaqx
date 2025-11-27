# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from abc import abstractmethod, ABC
import torch
import torch.nn.functional as F


class Loss(ABC):
    """Abstract base class for logit-energy matching loss functions.

    Loss functions in GQE compare the model's logits (predictions) with
    computed energies to guide the model toward selecting operator
    sequences that minimize energy.
    """

    @abstractmethod
    def compute(
            self,
            energies,
            gate_logits,
            gate_indices,
            log_values,
            **kwargs):
        pass


class ExpLogitMatching(Loss):
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

    def __init__(self, energy_offset) -> None:
        self.energy_offset = energy_offset
        self.loss_fn = torch.nn.MSELoss()

    def compute(
            self,
            energies,
            gate_logits,
            gate_indices,
            log_values,
            **kwargs):
        logits_tensor = torch.gather(
            gate_logits, 2, gate_indices.unsqueeze(-1)).squeeze(-1)
        mean_logits = torch.mean(logits_tensor, 1)
        log_values["mean_logits"] = torch.mean(
            mean_logits - self.energy_offset)
        mean_logits = torch.mean(logits_tensor, 1)
        device = mean_logits.device
        return self.loss_fn(
            torch.exp(-mean_logits),
            torch.exp(-energies.to(device) - self.energy_offset))


class GFlowLogitMatching(Loss):
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

    def __init__(self, energy_offset) -> None:
        self.loss_fn = torch.nn.MSELoss()
        self.energy_offset = energy_offset
        self.normalization = 10**-5
        self.param = torch.nn.Parameter(torch.tensor([0.0]))

    def compute(
            self,
            energies,
            gate_logits,
            gate_indices,
            log_values,
            **kwargs):
        logits_tensor = torch.gather(
            gate_logits, 2, gate_indices.unsqueeze(-1)).squeeze(-1)
        mean_logits = torch.mean(logits_tensor, 1)
        energy_offset = self.energy_offset + self.param / self.normalization
        log_values["energy_offset"] = energy_offset
        log_values["mean_logits"] = torch.mean(mean_logits - energy_offset)
        mean_logits = torch.mean(logits_tensor, 1)
        device = mean_logits.device
        loss = self.loss_fn(
            torch.exp(-mean_logits),
            torch.exp(-(energies.to(device) + energy_offset.to(device))))
        return loss


class GRPOLoss(Loss):
    """Generalized-RPO / clipped-PPO variant used in the original code."""

    def __init__(self, clip_ratio: float = 0.2):
        self.clip_ratio = clip_ratio
        self.old_log_probs = None
        self.advantages = None

    def compute(
            self,
            energies,
            gate_logits,
            gate_indices,
            log_values=None,
            **kwargs):
        current_log_probs = self.log_prob(
            gate_indices, gate_logits, kwargs["inverse_temperature"]
        )

        # nagative log likelihood loss
        win_id = torch.argmin(energies)
        log_prob_sum_win = torch.mean(current_log_probs[win_id])
        loss = -log_prob_sum_win

        # If all the generated circuits are identical, we use the inverse log
        # probability as the loss.
        if torch.std(energies) == 0:
            return loss

        # use the log probability from the first epoch as the reference.
        if kwargs["current_step"] == 0:
            self.old_log_probs = current_log_probs.detach()
            self.advantages = self.calc_advantage(energies)
            clipped_ratio = 1
        else:
            ratio = torch.exp(current_log_probs - self.old_log_probs)
            clipped_ratio = torch.clamp(
                ratio, 1. - self.clip_ratio, 1. + self.clip_ratio)

        loss -= (clipped_ratio * self.advantages.unsqueeze(1)).mean()
        return loss

    def calc_advantage(self, energies):
        return (energies.mean() - energies) / (energies.std() + 1e-8)

    def log_prob(
            self,
            gate_seqs,
            gate_logits,
            inverse_temperature):
        log_probs = torch.gather(
            F.log_softmax(-inverse_temperature * gate_logits, dim=-1),
            2,
            gate_seqs.unsqueeze(-1)
        ).squeeze(-1)
        return log_probs
