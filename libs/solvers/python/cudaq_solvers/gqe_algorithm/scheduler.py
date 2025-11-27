# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from abc import ABC, abstractmethod
import math


class TemperatureScheduler(ABC):
    """Abstract base class for temperature scheduling in GQE.
    
    Temperature scheduling controls how the temperature parameter changes during training,
    which affects the exploration vs exploitation trade-off in operator selection.
    """

    @abstractmethod
    def get_inverse_temperature(self):
        """Get current inverse temperature value.
        
        Returns:
            float: Current inverse temperature (beta)
        """
        pass
    
    @abstractmethod
    def update(self, **kwargs):
        """Update scheduler state.
        
        This can be used to adjust temperature dynamically based on training progress.
        
        Args:
            **kwargs: Optional keyword arguments (e.g., energies, loss, iteration)
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
        self.current_temperature = start

    def get_inverse_temperature(self):
        """Get current inverse temperature value.
        
        Returns:
            float: Current inverse temperature (beta)
        """
        return self.current_temperature

    def update(self, **kwargs):
        """Update temperature by incrementing by delta.
        
        Args:
            **kwargs: Unused, but accepts any keyword arguments for interface compatibility
        """
        self.current_temperature += self.delta


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
        self.current_iter = 0
        self.current_temperature = (maximum + minimum) / 2

    def get_inverse_temperature(self):
        """Get current inverse temperature value.
        
        Returns:
            float: Current inverse temperature (beta)
        """
        return self.current_temperature

    def update(self, **kwargs):
        """Update temperature following cosine schedule.
        
        Args:
            **kwargs: Unused, but accepts any keyword arguments for interface compatibility
        """
        self.current_iter += 1
        self.current_temperature = (self.maximum + self.minimum) / 2 - (
            self.maximum - self.minimum) / 2 * math.cos(
                2 * math.pi * self.current_iter / self.frequency)
