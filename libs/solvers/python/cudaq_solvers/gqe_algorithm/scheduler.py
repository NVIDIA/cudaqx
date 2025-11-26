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

