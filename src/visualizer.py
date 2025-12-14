"""
Visualization module for displaying the game state.
"""

import os
from typing import Optional, Union
from environment import Environment, Move


class GameVisualizer:
    """
    Handles visualization of the game state in the terminal.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.last_display = ""
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def display(self, 
                env: Environment, 
                step: int,
                pacman_id: str,
                ghost_id: str,
                pacman_move: Optional[Union[Move, tuple]] = None,
                ghost_move: Optional[Move] = None,
                result: Optional[str] = None):
        """
        Display the current game state.
        
        Args:
            env: The game environment
            step: Current step number
            pacman_id: Pacman student ID
            ghost_id: Ghost student ID
            pacman_move: Last move made by Pacman
            ghost_move: Last move made by Ghost
            result: Game result if finished
        """
        self.clear_screen()
        
        # Header
        print(f"\n{'='*60}")
        print(f"{'PACMAN vs GHOST ARENA':^60}")
        print(f"{'='*60}\n")
        
        # Player info
        print(f"🔵 Pacman (P): {pacman_id:20} ", end="")
        if pacman_move:
            print(f"Last move: {self._format_move(pacman_move)}", end="")
        print()
        
        print(f"🔴 Ghost  (G): {ghost_id:20} ", end="")
        if ghost_move:
            print(f"Last move: {self._format_move(ghost_move)}", end="")
        print()
        
        print(f"\nStep: {step}/{env.max_steps}")
        
        distance = env.get_distance(env.pacman_pos, env.ghost_pos)
        print(f"Distance: {distance} cells")
        
        print(f"\n{'─'*60}\n")
        
        # Map
        map_display = env.render()
        
        # Add color codes for better visibility (if terminal supports it)
        map_display = map_display.replace('P', '\033[94mP\033[0m')  # Blue
        map_display = map_display.replace('G', '\033[91mG\033[0m')  # Red
        map_display = map_display.replace('X', '\033[93mX\033[0m')  # Yellow (collision)
        
        print(map_display)
        
        print(f"\n{'─'*60}")
        
        # Result if game is over
        if result:
            print()
            if result == 'pacman_wins':
                print(f"{'🏆 PACMAN WINS! 🏆':^60}")
            elif result == 'ghost_wins':
                print(f"{'🏆 GHOST WINS! 🏆':^60}")
            elif result == 'draw':
                print(f"{'🤝 DRAW! 🤝':^60}")
            print(f"{'─'*60}")
        
        print()
    
    def display_error(self, error_msg: str, agent_type: str, student_id: str):
        """
        Display an error message.
        
        Args:
            error_msg: The error message
            agent_type: Type of agent that caused the error
            student_id: Student ID
        """
        print(f"\n{'!'*60}")
        print(f"ERROR in {agent_type.upper()} agent ({student_id}):")
        print(f"{error_msg}")
        print(f"{'!'*60}\n")

    def _format_move(self, move) -> str:
        if isinstance(move, tuple) and len(move) == 2:
            base_move, steps = move
            label = base_move.name if isinstance(base_move, Move) else str(base_move)
            steps = int(steps)
            if steps > 1:
                return f"{label} x{steps}"
            return label
        if isinstance(move, Move):
            return move.name
        return str(move)
