"""
Agent interface definition for students to implement.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
from environment import Move


class AgentInterface(ABC):
    """
    Base interface that all student agents must implement.
    
    Agents are stateful and may accumulate memory across steps.
    """
    
    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the agent.
        Students can use this to set up any data structures they need.
        
        Args:
            **kwargs: Optional arguments for agent configuration
        """
        pass
    
    @abstractmethod
    def step(self, map_state: np.ndarray, 
             my_position: Tuple[int, int], 
             enemy_position: Optional[Tuple[int, int]],
             step_number: int) -> Move:
        """
        Decide the next move based on the current environment.
        
        Args:
            map_state: 2D numpy array where:
                - 1 = wall (always visible)
                - 0 = empty space (visible)
                - -1 = unseen/fog (when limited observation is enabled)
            my_position: Current position as (row, col) in absolute coordinates
            enemy_position: Enemy's position as (row, col) if visible, 
                None if outside observation range
            step_number: Current step number in the game
            
        Returns:
            For Pacman agents: Either a Move enum value or a tuple of
            (Move, steps) where steps is an integer between 1 and the
            configured maximum straight-line speed.
            For Ghost agents: Move enum value (UP, DOWN, LEFT, RIGHT, or STAY)
        """
        pass


class PacmanAgent(AgentInterface):
    """
    Interface for Pacman agents (seekers).
    Goal: Catch the ghost.
    """
    pass


class GhostAgent(AgentInterface):
    """
    Interface for Ghost agents (hiders).
    Goal: Avoid being caught by Pacman.
    """
    pass
