"""
Environment module for the Pacman vs Ghost arena.
Defines the game map, positions, and rules.
"""

import numpy as np
from typing import Tuple, List, Optional, Set
from enum import Enum


class CellType(Enum):
    """Types of cells in the game map."""
    UNSEEN = -1  # Fog of war marker for limited observation
    EMPTY = 0
    WALL = 1
    PACMAN = 2
    GHOST = 3


class Move(Enum):
    """Valid moves for agents."""
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    STAY = (0, 0)


class Environment:
    """
    Game environment that manages the map and agent positions.
    """
    
    def __init__(
        self,
        map_layout: Optional[np.ndarray] = None,
        max_steps: int = 200,
        deterministic_starts: bool = True,
        capture_distance_threshold: int = 1,
        pacman_speed: int = 1
    ):
        """
        Initialize the environment.
        
        Args:
            map_layout: 2D numpy array where 1 = wall, 0 = empty
            max_steps: Maximum number of steps before game ends in a draw
        """
        self.default_pacman_start = None
        self.default_ghost_start = None
        self.deterministic_starts = deterministic_starts
        self.capture_distance_threshold = max(1, int(capture_distance_threshold))
        self.pacman_speed = max(1, int(pacman_speed))

        if map_layout is None:
            # Default classic Pacman-style map
            self.map = self._create_default_map()
        else:
            self.map = map_layout.copy()
        
        self.height, self.width = self.map.shape
        self.max_steps = max_steps
        self.current_step = 0
        
        # Initialize positions
        self.pacman_pos = None
        self.ghost_pos = None
        self.reset()
    
    def _create_default_map(self) -> np.ndarray:
        """
        Create a default Pacman-style map.
        1 = wall, 0 = empty space
        """
        layout = [
            "#####################",
            "#.........#.........#",
            "#.###.###.#.###.###.#",
            "#...................#",
            "#.###.#.#####.#.###.#",
            "#.....#...#...#.....#",
            "#####.### # ###.#####",
            "    #.#       #.#    ",
            "#####.# ##-## #.#####",
            "     .  . G .  .     ",
            "#####.# ##### #.#####",
            "    #.#       #.#    ",
            "#####.# ##### #.#####",
            "#.........#.........#",
            "#.###.###.#.###.###.#",
            "#...#.....P.....#...#",
            "###.#.#.#####.#.#.###",
            "#.....#...#...#.....#",
            "#.#######.#.#######.#",
            "#...................#",
            "#####################"
        ]
        
        pacman_start = None
        ghost_start = None
        map_array = np.zeros((len(layout), len(layout[0])), dtype=int)
        for i, row in enumerate(layout):
            for j, cell in enumerate(row):
                if cell == '#':
                    map_array[i, j] = 1
                elif cell == '-':
                    map_array[i, j] = 1
                else:
                    map_array[i, j] = 0
                    if cell == 'P':
                        pacman_start = (i, j)
                    elif cell == 'G':
                        ghost_start = (i, j)

        self.default_pacman_start = pacman_start
        self.default_ghost_start = ghost_start
        return map_array
    
    def reset(self) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Reset the environment to initial state.
        
        Returns:
            Tuple of (map, pacman_position, ghost_position)
        """
        self.current_step = 0
        empty_cells = None
        if (not self.deterministic_starts or
                self.default_pacman_start is None or
                self.default_ghost_start is None):
            empty_cells = np.argwhere(self.map == 0)

        if self.deterministic_starts and self.default_pacman_start is not None:
            self.pacman_pos = tuple(int(v) for v in self.default_pacman_start)
        else:
            bottom_cells = empty_cells[empty_cells[:, 0] > self.height * 0.6]
            if len(bottom_cells) > 0:
                pacman_idx = np.random.choice(len(bottom_cells))
                self.pacman_pos = tuple(bottom_cells[pacman_idx])
            else:
                self.pacman_pos = tuple(empty_cells[0])

        if self.deterministic_starts and self.default_ghost_start is not None:
            self.ghost_pos = tuple(int(v) for v in self.default_ghost_start)
        else:
            top_cells = empty_cells[empty_cells[:, 0] < self.height * 0.4]
            if len(top_cells) > 0:
                ghost_idx = np.random.choice(len(top_cells))
                self.ghost_pos = tuple(top_cells[ghost_idx])
            else:
                self.ghost_pos = tuple(empty_cells[-1])
        
        return self.get_state()
    
    def get_state(self) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Get the current state of the environment.
        
        Returns:
            Tuple of (map, pacman_position, ghost_position)
        """
        return self.map.copy(), self.pacman_pos, self.ghost_pos
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if a position is valid (within bounds and not a wall).
        
        Args:
            pos: (row, col) position
            
        Returns:
            True if position is valid, False otherwise
        """
        row, col = pos
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        return self.map[row, col] == 0
    
    def apply_move(self, current_pos: Tuple[int, int], move: Move) -> Tuple[int, int]:
        """
        Apply a move to a position.
        
        Args:
            current_pos: Current (row, col) position
            move: Move to apply
            
        Returns:
            New position after move (stays same if invalid)
        """
        delta_row, delta_col = move.value
        new_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)
        
        if self.is_valid_position(new_pos):
            return new_pos
        return current_pos

    def _apply_pacman_move(self, current_pos: Tuple[int, int], move: Move, steps: int) -> Tuple[int, int]:
        if move == Move.STAY:
            return current_pos

        new_pos = current_pos
        for _ in range(steps):
            candidate = self.apply_move(new_pos, move)
            if candidate == new_pos:
                break
            new_pos = candidate
        return new_pos
    
    def step(self, pacman_move: Move, ghost_move: Move) -> Tuple[bool, str, Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]]:
        """
        Execute one step of the game.
        
        Args:
            pacman_move: Move chosen by Pacman agent
            ghost_move: Move chosen by Ghost agent
            
        Returns:
            Tuple of (game_over, result, new_state)
            - game_over: True if game has ended
            - result: 'pacman_wins', 'ghost_wins', or 'draw'
            - new_state: New state of the environment
        """
        self.current_step += 1
        
        # Apply moves
        pacman_move, requested_steps = self._normalize_pacman_action(pacman_move)
        new_pacman_pos = self._apply_pacman_move(self.pacman_pos, pacman_move, requested_steps)
        new_ghost_pos = self.apply_move(self.ghost_pos, ghost_move)
        
        # Update positions
        self.pacman_pos = new_pacman_pos
        self.ghost_pos = new_ghost_pos
        
        # Check win conditions
        distance = self.get_distance(self.pacman_pos, self.ghost_pos)
        if distance < self.capture_distance_threshold:
            return True, 'pacman_wins', self.get_state()
        
        # Ghost wins if Pacman fails to catch within the allotted steps
        if self.current_step >= self.max_steps:
            return True, 'ghost_wins', self.get_state()
        
        return False, '', self.get_state()
    
    def get_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate Manhattan distance between two positions.
        
        Args:
            pos1: First position (row, col)
            pos2: Second position (row, col)
            
        Returns:
            Manhattan distance
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def render(self) -> str:
        """
        Render the current state as a string.
        
        Returns:
            String representation of the map with agents
        """
        display = self.map.copy().astype(str)
        display[display == '0'] = '.'
        display[display == '1'] = '#'
        
        # Mark agent positions
        if self.pacman_pos:
            display[self.pacman_pos] = 'P'
        if self.ghost_pos:
            display[self.ghost_pos] = 'G'
        
        # If they're at the same position, show collision
        if self.pacman_pos == self.ghost_pos:
            display[self.pacman_pos] = 'X'
        
        rows = [''.join(row) for row in display]
        return '\n'.join(rows)

    def _normalize_pacman_action(self, action) -> Tuple[Move, int]:
        if isinstance(action, Move):
            move = action
            steps = 1
        elif isinstance(action, tuple) and len(action) == 2:
            move, steps = action
        else:
            raise ValueError("Pacman action must be Move or (Move, steps) tuple")

        if not isinstance(move, Move):
            raise ValueError("Pacman action must include a Move enum")

        try:
            steps = int(steps)
        except (TypeError, ValueError):
            steps = 1

        if steps < 1:
            steps = 1
        steps = min(steps, self.pacman_speed)
        return move, steps

    def _in_bounds(self, row: int, col: int) -> bool:
        """Check if coordinates are within map bounds."""
        return 0 <= row < self.height and 0 <= col < self.width

    def get_visible_cells_cross(
        self,
        observer_pos: Tuple[int, int],
        radius: int
    ) -> Set[Tuple[int, int]]:
        """
        Return set of (row, col) cells visible via direct cross.
        
        Rays extend up to `radius` tiles in each cardinal direction,
        stopping at walls (non-obscured line of sight).
        
        Args:
            observer_pos: Position of the observing agent
            radius: Maximum observation distance in each direction
            
        Returns:
            Set of visible cell coordinates
        """
        visible: Set[Tuple[int, int]] = {observer_pos}
        row, col = observer_pos

        # Cast rays in all 4 cardinal directions
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            for dist in range(1, radius + 1):
                nr, nc = row + dr * dist, col + dc * dist
                if not self._in_bounds(nr, nc):
                    break
                if self.map[nr, nc] == 1:  # Wall stops the ray
                    break
                visible.add((nr, nc))

        return visible

    def get_observation(
        self,
        agent_type: str,
        pacman_radius: int,
        ghost_radius: int
    ) -> Tuple[np.ndarray, Tuple[int, int], Optional[Tuple[int, int]]]:
        """
        Return observation for the specified agent.
        
        If radius is 0, returns full visibility (current behavior).
        Otherwise, cells outside observation are marked as -1 (UNSEEN).
        Walls are always visible (structural map knowledge).
        Enemy position is None if not within visible cells.
        
        Args:
            agent_type: 'pacman' or 'ghost'
            pacman_radius: Observation radius for Pacman (0 = full visibility)
            ghost_radius: Observation radius for Ghost (0 = full visibility)
            
        Returns:
            Tuple of (observation_array, my_position, enemy_position_or_none)
        """
        if agent_type == 'pacman':
            my_pos = self.pacman_pos
            enemy_pos = self.ghost_pos
            radius = pacman_radius
        else:
            my_pos = self.ghost_pos
            enemy_pos = self.pacman_pos
            radius = ghost_radius

        # Full visibility mode (radius 0 or unspecified)
        if radius <= 0:
            return self.map.copy(), my_pos, enemy_pos

        # Limited visibility mode
        visible_cells = self.get_visible_cells_cross(my_pos, radius)

        # Start with map copy (walls always known)
        obs = self.map.copy()

        # Mark non-visible empty cells as UNSEEN (-1)
        for r in range(self.height):
            for c in range(self.width):
                if (r, c) not in visible_cells and obs[r, c] == 0:
                    obs[r, c] = -1

        # Enemy visible only if in visible cells
        visible_enemy = enemy_pos if enemy_pos in visible_cells else None

        return obs, my_pos, visible_enemy
