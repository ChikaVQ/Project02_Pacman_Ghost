"""
Module for loading student agents dynamically.
"""

import sys
import importlib.util
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING
from agent_interface import PacmanAgent, GhostAgent

if TYPE_CHECKING:
    from environment import Move


class AgentLoadError(Exception):
    """Exception raised when agent loading fails."""
    pass


class AgentLoader:
    """
    Loads student agents from the submissions directory.
    """
    
    def __init__(self, submissions_dir: str = "submissions"):
        """
        Initialize the agent loader.
        
        Args:
            submissions_dir: Directory containing student submissions
        """
        self.submissions_dir = Path(submissions_dir)
        if not self.submissions_dir.exists():
            self.submissions_dir.mkdir(parents=True)
    
    def load_agent(self, student_id: str, agent_type: str, init_kwargs: Optional[dict] = None) -> object:
        """
        Load a student's agent.
        
        Args:
            student_id: Student ID (folder name in submissions/)
            agent_type: 'pacman' or 'ghost'
            
        Returns:
            Instantiated agent object
            
        Raises:
            AgentLoadError: If agent cannot be loaded or doesn't meet requirements
        """
        agent_dir = self.submissions_dir / student_id
        agent_file = agent_dir / "agent.py"
        
        # Check if agent file exists
        if not agent_file.exists():
            raise AgentLoadError(
                f"Agent file not found for student {student_id} at {agent_file}"
            )
        
        # Load the module
        try:
            spec = importlib.util.spec_from_file_location(
                f"{student_id}.agent", 
                agent_file
            )
            module = importlib.util.module_from_spec(spec)
            
            # Add the agent directory to sys.path temporarily
            # so students can import their own modules
            agent_dir_str = str(agent_dir.absolute())
            if agent_dir_str not in sys.path:
                sys.path.insert(0, agent_dir_str)
            
            spec.loader.exec_module(module)
            
        except Exception as e:
            raise AgentLoadError(
                f"Failed to load module for student {student_id}: {str(e)}"
            )
        
        # Get the appropriate agent class
        if agent_type.lower() == 'pacman':
            if not hasattr(module, 'PacmanAgent'):
                raise AgentLoadError(
                    f"Student {student_id}'s agent.py must define a 'PacmanAgent' class"
                )
            agent_class = module.PacmanAgent
            expected_parent = PacmanAgent
        elif agent_type.lower() == 'ghost':
            if not hasattr(module, 'GhostAgent'):
                raise AgentLoadError(
                    f"Student {student_id}'s agent.py must define a 'GhostAgent' class"
                )
            agent_class = module.GhostAgent
            expected_parent = GhostAgent
        else:
            raise AgentLoadError(f"Invalid agent type: {agent_type}")
        
        # Verify the agent has required methods
        required_methods = ['__init__', 'step']
        for method in required_methods:
            if not hasattr(agent_class, method):
                raise AgentLoadError(
                    f"Agent class must implement '{method}' method"
                )
        
        # Instantiate the agent
        try:
            kwargs = init_kwargs or {}
            agent_instance = agent_class(**kwargs)
        except Exception as e:
            raise AgentLoadError(
                f"Failed to instantiate agent for student {student_id}: {str(e)}"
            )
        
        return agent_instance
    
    def validate_agent_move(self, move, agent_type: str, student_id: str, pacman_speed: Optional[int] = None):
        """
        Validate that an agent's move is legal.
        
        Args:
            move: The move returned by the agent
            agent_type: 'pacman' or 'ghost'
            student_id: Student ID for error messages
            
        Raises:
            AgentLoadError: If move is invalid
        """
        from environment import Move
        
        if agent_type.lower() == 'pacman':
            return self._validate_pacman_action(move, student_id, pacman_speed)

        if not isinstance(move, Move):
            raise AgentLoadError(
                f"Agent {student_id} ({agent_type}) returned invalid move type: {type(move)}. "
                f"Must return a Move enum value."
            )
        return move

    def _validate_pacman_action(self, action, student_id: str, pacman_speed: Optional[int]) -> Tuple['Move', int]:
        from environment import Move

        if isinstance(action, Move):
            move = action
            steps = 1
        elif isinstance(action, tuple) and len(action) == 2:
            move, steps = action
        else:
            raise AgentLoadError(
                f"Agent {student_id} (pacman) must return a Move or a (Move, steps) tuple. "
                f"Got {action!r}."
            )

        if not isinstance(move, Move):
            raise AgentLoadError(
                f"Agent {student_id} (pacman) returned invalid move component: {move!r}."
            )

        try:
            steps = int(steps)
        except (TypeError, ValueError):
            raise AgentLoadError(
                f"Agent {student_id} (pacman) provided non-integer steps value: {steps!r}."
            )

        if steps < 1:
            raise AgentLoadError(
                f"Agent {student_id} (pacman) must request at least 1 step."
            )

        if pacman_speed is not None and steps > pacman_speed:
            raise AgentLoadError(
                f"Agent {student_id} (pacman) requested {steps} steps which exceeds the maximum speed {pacman_speed}."
            )

        return move, steps
