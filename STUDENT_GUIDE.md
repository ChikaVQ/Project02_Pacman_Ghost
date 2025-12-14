# Pacman vs Ghost Arena - Student Guide

Welcome to the Pacman vs Ghost Arena! This guide will help you implement your own AI agents using search algorithms.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Understanding the Game](#understanding-the-game)
4. [Creating Your Agent](#creating-your-agent)
5. [Implementing Search Algorithms](#implementing-search-algorithms)
6. [Testing Your Agent](#testing-your-agent)
7. [Debugging Tips](#debugging-tips)
8. [Common Errors](#common-errors)
9. [Advanced Strategies](#advanced-strategies)

---

## Quick Start

### 1. Create Your Submission Folder

```bash
cd submissions
mkdir <your_student_id>
```

Replace `<your_student_id>` with your actual student ID (e.g., `student_001`, `alice`, `john_doe`)

### 2. Copy the Template

```bash
cp TEMPLATE_agent.py <your_student_id>/agent.py
```

### 3. Edit Your Agent

Open `submissions/<your_student_id>/agent.py` in your favorite editor and implement your search algorithm.

### 4. Test Your Agent

```bash
cd ../src
python arena.py --seek <your_student_id> --hide example_student
```

---

## Installation

### Prerequisites

- **Python 3.7+** (Python 3.11 recommended)
- **Conda** environment manager
- **NumPy** library

### Setup Steps

```bash
# 1. Activate conda environment
conda activate ml

# 2. Install dependencies
pip install -r requirements.txt
```

The `requirements.txt` contains:
```text
numpy>=1.20.0
```

### Verify Installation

Test that everything works:

```bash
cd src
python arena.py --seek example_student --hide example_student
```

You should see a colorful visualization of Pacman (blue) chasing Ghost (red) in a maze!

---

## Understanding the Game

### Objective

- **Pacman (Seeker)**: Catch the Ghost by moving to the same position
- **Ghost (Hider)**: Evade Pacman for as long as possible (survive until max steps)

### Win Conditions

- **Pacman wins**: Catches Ghost (reaches same position)
- **Ghost wins**: Survives for max steps (default: 200) without being caught
- **Draw**: Currently treated as Ghost win

### The Map

The game is played on a 21×21 grid maze:

- `0` = Empty space (you can move here)
- `1` = Wall (you cannot move here)
- The maze has a classic Pacman layout with corridors and walls

### Movement

You can move in 5 directions:

```python
Move.UP      # Move up    (row - 1, col)
Move.DOWN    # Move down  (row + 1, col)
Move.LEFT    # Move left  (row, col - 1)
Move.RIGHT   # Move right (row, col + 1)
Move.STAY    # Don't move (row, col)
```

### Important: Synchronous Execution

**Both agents move at the SAME time!**

- Both receive the same state
- Both decide their moves simultaneously
- Both positions update at once

This means you cannot react to your opponent's move - you must predict it!

### Game Information You Receive

Every step, your `step()` method receives:

1. **`map_state`**: 2D numpy array of the maze
2. **`my_position`**: Your current position as `(row, col)`
3. **`enemy_position`**: Enemy's current position as `(row, col)`
4. **`step_number`**: Current step number (starts at 1)

---

## Creating Your Agent

### Required Code Structure

Your `agent.py` must define a `PacmanAgent` and/or `GhostAgent` class:

```python
import sys
from pathlib import Path

# Add src to path so you can import framework classes
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


class PacmanAgent(BasePacmanAgent):
    """Your Pacman (Seeker) implementation."""
    
    def __init__(self, **kwargs):
        """Initialize your agent. Called once at game start."""
        super().__init__(**kwargs)
        # Initialize your data structures here
        # Example: self.path_cache = {}
        
    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Called every step. Return your move decision.
        
        Args:
            map_state: numpy array (height x width), 0=empty, 1=wall
            my_position: (row, col) tuple of Pacman's position
            enemy_position: (row, col) tuple of Ghost's position
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your search algorithm here!
        
        # Example: Simple greedy move toward enemy
        row_diff = enemy_position[0] - my_position[0]
        col_diff = enemy_position[1] - my_position[1]
        
        if abs(row_diff) > abs(col_diff):
            if row_diff > 0:
                return Move.DOWN
            else:
                return Move.UP
        else:
            if col_diff > 0:
                return Move.RIGHT
            else:
                return Move.LEFT


class GhostAgent(BaseGhostAgent):
    """Your Ghost (Hider) implementation."""
    
    def __init__(self, **kwargs):
        """Initialize your agent. Called once at game start."""
        super().__init__(**kwargs)
        # Initialize your data structures here
        
    def step(self, map_state, my_position, enemy_position, step_number):
        """
        Called every step. Return your move decision.
        
        Args:
            map_state: numpy array (height x width), 0=empty, 1=wall
            my_position: (row, col) tuple of Ghost's position
            enemy_position: (row, col) tuple of Pacman's position
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        # TODO: Implement your evasion algorithm here!
        
        # Example: Simple greedy move away from enemy
        row_diff = enemy_position[0] - my_position[0]
        col_diff = enemy_position[1] - my_position[1]
        
        if abs(row_diff) > abs(col_diff):
            if row_diff > 0:
                return Move.UP  # Move away
            else:
                return Move.DOWN
        else:
            if col_diff > 0:
                return Move.LEFT  # Move away
            else:
                return Move.RIGHT
```

### Required Imports

```python
from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
```

### Essential Helper Functions

Add these helper methods to your agent class:

```python
def _is_valid_position(self, pos, map_state):
    """Check if position is valid (not wall, within bounds)."""
    row, col = pos
    height, width = map_state.shape
    
    # Check bounds
    if row < 0 or row >= height or col < 0 or col >= width:
        return False
    
    # Check not a wall
    return map_state[row, col] == 0


def _apply_move(self, pos, move):
    """Apply a move to a position, return new position."""
    delta_row, delta_col = move.value
    return (pos[0] + delta_row, pos[1] + delta_col)


def _get_neighbors(self, pos, map_state):
    """Get all valid neighboring positions and their moves."""
    neighbors = []
    
    for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
        next_pos = self._apply_move(pos, move)
        if self._is_valid_position(next_pos, map_state):
            neighbors.append((next_pos, move))
    
    return neighbors


def _manhattan_distance(self, pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
```

---

## Implementing Search Algorithms

### Breadth-First Search (BFS)

**Best for Pacman** - Finds shortest path to Ghost

```python
def bfs(self, start, goal, map_state):
    """
    Find shortest path from start to goal using BFS.
    
    Returns:
        List of Move enums representing the path, or [Move.STAY] if no path
    """
    from collections import deque
    
    # Queue stores (position, path_to_reach_it)
    queue = deque([(start, [])])
    visited = {start}
    
    while queue:
        current_pos, path = queue.popleft()
        
        # Found the goal!
        if current_pos == goal:
            return path
        
        # Explore neighbors
        for next_pos, move in self._get_neighbors(current_pos, map_state):
            if next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, path + [move]))
    
    # No path found
    return [Move.STAY]
```

**Usage in step():**

```python
def step(self, map_state, my_position, enemy_position, step_number):
    path = self.bfs(my_position, enemy_position, map_state)
    if path:
        return path[0]  # Return first move in path
    return Move.STAY
```

### A* Search

**Best for Pacman** - Optimal path with heuristic guidance

```python
def astar(self, start, goal, map_state):
    """
    Find optimal path from start to goal using A* search.
    
    Returns:
        List of Move enums representing the path, or [Move.STAY] if no path
    """
    from heapq import heappush, heappop
    
    def heuristic(pos):
        """Manhattan distance heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    # Priority queue stores (f_cost, position, path)
    # f_cost = g_cost + h_cost
    frontier = [(0, start, [])]
    visited = set()
    
    while frontier:
        f_cost, current_pos, path = heappop(frontier)
        
        # Found the goal!
        if current_pos == goal:
            return path
        
        # Skip if already visited
        if current_pos in visited:
            continue
        
        visited.add(current_pos)
        
        # Explore neighbors
        for next_pos, move in self._get_neighbors(current_pos, map_state):
            if next_pos not in visited:
                new_path = path + [move]
                g_cost = len(new_path)  # Cost so far
                h_cost = heuristic(next_pos)  # Estimated cost to goal
                f_cost = g_cost + h_cost  # Total estimated cost
                heappush(frontier, (f_cost, next_pos, new_path))
    
    # No path found
    return [Move.STAY]
```

**Usage in step():**

```python
def step(self, map_state, my_position, enemy_position, step_number):
    path = self.astar(my_position, enemy_position, map_state)
    if path:
        return path[0]
    return Move.STAY
```

### Greedy Best-First Search

**Faster than A*** but not always optimal

```python
def greedy_best_first(self, start, goal, map_state):
    """
    Find path using greedy best-first search (only heuristic, no path cost).
    
    Returns:
        List of Move enums, or [Move.STAY] if no path
    """
    from heapq import heappush, heappop
    
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    frontier = [(heuristic(start), start, [])]
    visited = set()
    
    while frontier:
        _, current_pos, path = heappop(frontier)
        
        if current_pos == goal:
            return path
        
        if current_pos in visited:
            continue
        
        visited.add(current_pos)
        
        for next_pos, move in self._get_neighbors(current_pos, map_state):
            if next_pos not in visited:
                new_path = path + [move]
                heappush(frontier, (heuristic(next_pos), next_pos, new_path))
    
    return [Move.STAY]
```

### Evasion Strategy for Ghost

**Maximize distance from Pacman:**

```python
def find_escape_move(self, my_position, enemy_position, map_state):
    """
    Find move that maximizes distance from enemy.
    
    Returns:
        Move enum
    """
    best_move = Move.STAY
    best_distance = self._manhattan_distance(my_position, enemy_position)
    
    for next_pos, move in self._get_neighbors(my_position, map_state):
        distance = self._manhattan_distance(next_pos, enemy_position)
        if distance > best_distance:
            best_distance = distance
            best_move = move
    
    return best_move
```

**Advanced: Find furthest reachable position**

```python
def find_furthest_position(self, my_position, enemy_position, map_state):
    """
    Use BFS from enemy's position to find furthest reachable point.
    Then navigate toward it.
    
    Returns:
        Move enum to move toward furthest position
    """
    from collections import deque
    
    # BFS from enemy to find distances
    queue = deque([(enemy_position, 0)])
    visited = {enemy_position: 0}
    
    while queue:
        current, dist = queue.popleft()
        
        for next_pos, _ in self._get_neighbors(current, map_state):
            if next_pos not in visited:
                visited[next_pos] = dist + 1
                queue.append((next_pos, dist + 1))
    
    # Find maximum distance among reachable positions
    max_dist = -1
    best_positions = []
    
    for pos, dist in visited.items():
        if dist > max_dist:
            max_dist = dist
            best_positions = [pos]
        elif dist == max_dist:
            best_positions.append(pos)
    
    # Navigate to closest of the furthest positions
    if best_positions:
        # Find path to nearest best position
        best_target = min(best_positions, 
                         key=lambda p: self._manhattan_distance(my_position, p))
        path = self.bfs(my_position, best_target, map_state)
        if path:
            return path[0]
    
    return Move.STAY
```

---

## Testing Your Agent

### Basic Testing

```bash
# From src directory
cd src

# Test your Pacman against example Ghost
python arena.py --seek <your_id> --hide example_student

# Test your Ghost against example Pacman
python arena.py --seek example_student --hide <your_id>

# Test both your agents against each other
python arena.py --seek <your_id> --hide <your_id>
```

### Testing Options

```bash
# Faster testing (no visualization)
python arena.py --seek <your_id> --hide example_student --no-viz

# Slower visualization for debugging
python arena.py --seek <your_id> --hide example_student --delay 0.5

# Longer game
python arena.py --seek <your_id> --hide example_student --max-steps 300

# Shorter game (faster testing)
python arena.py --seek <your_id> --hide example_student --max-steps 50
```

### Using the Run Script

From the Arena directory:

```bash
./run_game.sh --seek <your_id> --hide example_student
```

---

## Debugging Tips

### 1. Add Print Statements

```python
def step(self, map_state, my_position, enemy_position, step_number):
    print(f"Step {step_number}: My pos={my_position}, Enemy pos={enemy_position}")
    
    path = self.bfs(my_position, enemy_position, map_state)
    print(f"  Found path: {[m.name for m in path[:5]]}")  # First 5 moves
    
    if path:
        return path[0]
    return Move.STAY
```

### 2. Watch Visualization Slowly

```bash
python arena.py --seek <your_id> --hide example_student --delay 1.0
```

This gives you 1 second between moves to see what's happening.

### 3. Test Edge Cases

```python
# Test helper functions independently
agent = PacmanAgent()
test_pos = (10, 10)
test_map = np.zeros((21, 21))  # Empty map

# Test is_valid_position
assert agent._is_valid_position(test_pos, test_map) == True
assert agent._is_valid_position((-1, 10), test_map) == False

# Test apply_move
new_pos = agent._apply_move(test_pos, Move.UP)
assert new_pos == (9, 10)

print("All tests passed!")
```

### 4. Check for Infinite Loops

Make sure your search algorithms terminate:

```python
def bfs(self, start, goal, map_state):
    queue = deque([(start, [])])
    visited = {start}  # IMPORTANT: Track visited nodes!
    
    max_iterations = 10000  # Safety limit
    iterations = 0
    
    while queue and iterations < max_iterations:
        iterations += 1
        # ... rest of BFS
        
    if iterations >= max_iterations:
        print("WARNING: BFS hit iteration limit!")
    
    return [Move.STAY]
```

### 5. Validate Return Values

```python
def step(self, map_state, my_position, enemy_position, step_number):
    move = self.calculate_best_move(...)
    
    # Validate before returning
    if not isinstance(move, Move):
        print(f"ERROR: Invalid move type: {type(move)}")
        return Move.STAY
    
    return move
```

---

## Common Errors

### Error: "Agent file not found"

**Cause:** Folder name doesn't match the ID you used in command

**Solution:**
```bash
# Check your folder name
ls submissions/

# Make sure it matches exactly
python arena.py --seek exact_folder_name --hide example_student
```

### Error: "Must define a 'PacmanAgent' class"

**Cause:** Class name is wrong or misspelled

**Solution:**
- Class must be named **exactly** `PacmanAgent` or `GhostAgent`
- Check for typos: `PacMan`, `Pacman`, `pacmanAgent` are all WRONG
- Make sure class inherits: `class PacmanAgent(BasePacmanAgent):`

### Error: "Returned invalid move type"

**Cause:** Returning wrong type (string, tuple, None, etc.)

**Solution:**
```python
# ❌ WRONG
return "UP"
return (0, -1)
return 0
return None

# ✅ CORRECT
return Move.UP
return Move.DOWN
return Move.LEFT
return Move.RIGHT
return Move.STAY
```

### Error: Agent crashes with IndexError

**Cause:** Accessing map positions without validation

**Solution:**
```python
# Always validate before accessing
if self._is_valid_position(pos, map_state):
    value = map_state[pos[0], pos[1]]  # Safe
```

### Error: Agent takes too long / timeout

**Cause:** Inefficient algorithm or infinite loop

**Solution:**
1. Add visited set to prevent revisiting nodes
2. Limit search depth
3. Use better data structures (heap for A*, deque for BFS)
4. Add iteration limit for safety

---

## Advanced Strategies

### Strategy 1: Path Replanning

Don't recompute entire path every step:

```python
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_path = []
        self.last_enemy_pos = None
    
    def step(self, map_state, my_position, enemy_position, step_number):
        # Only replan if enemy moved significantly or path is empty
        if (not self.current_path or 
            self.last_enemy_pos is None or
            self._manhattan_distance(enemy_position, self.last_enemy_pos) > 3):
            
            self.current_path = self.astar(my_position, enemy_position, map_state)
            self.last_enemy_pos = enemy_position
        
        # Follow current path
        if self.current_path:
            next_move = self.current_path.pop(0)
            return next_move
        
        return Move.STAY
```

### Strategy 2: Predictive Movement (Advanced)

Predict where enemy will move:

```python
def predict_enemy_move(self, enemy_pos, my_pos, map_state):
    """Predict enemy's next move (assume they move away from us)."""
    best_move = Move.STAY
    best_distance = 0
    
    for next_pos, move in self._get_neighbors(enemy_pos, map_state):
        distance = self._manhattan_distance(next_pos, my_pos)
        if distance > best_distance:
            best_distance = distance
            best_move = move
    
    # Return predicted position
    return self._apply_move(enemy_pos, best_move)

def step(self, map_state, my_position, enemy_position, step_number):
    # Navigate to predicted position instead of current position
    predicted_enemy_pos = self.predict_enemy_move(
        enemy_position, my_position, map_state
    )
    
    path = self.astar(my_position, predicted_enemy_pos, map_state)
    
    if path:
        return path[0]
    return Move.STAY
```

### Strategy 3: Minimax for Ghost (Very Advanced)

```python
def minimax(self, state, depth, is_maximizing_player, map_state):
    """
    Minimax algorithm for Ghost (minimizing player).
    
    Args:
        state: (my_pos, enemy_pos)
        depth: Search depth remaining
        is_maximizing_player: True if Pacman's turn, False if Ghost's turn
        map_state: The maze map
        
    Returns:
        (best_score, best_move)
    """
    my_pos, enemy_pos = state
    
    # Base case: reached depth limit or caught
    if depth == 0 or my_pos == enemy_pos:
        # Return negative distance (Ghost wants to maximize distance)
        return -self._manhattan_distance(my_pos, enemy_pos), Move.STAY
    
    if is_maximizing_player:  # Pacman's turn (wants to minimize distance)
        best_score = float('-inf')
        best_move = Move.STAY
        
        for next_pos, move in self._get_neighbors(enemy_pos, map_state):
            new_state = (my_pos, next_pos)
            score, _ = self.minimax(new_state, depth-1, False, map_state)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_score, best_move
    
    else:  # Ghost's turn (wants to maximize distance)
        best_score = float('inf')
        best_move = Move.STAY
        
        for next_pos, move in self._get_neighbors(my_pos, map_state):
            new_state = (next_pos, enemy_pos)
            score, _ = self.minimax(new_state, depth-1, True, map_state)
            if score < best_score:
                best_score = score
                best_move = move
        
        return best_score, best_move

def step(self, map_state, my_position, enemy_position, step_number):
    _, best_move = self.minimax(
        (my_position, enemy_position),
        depth=3,  # Search 3 moves ahead
        is_maximizing_player=False,
        map_state=map_state
    )
    return best_move
```

---

## Quick Reference

### Available Moves

```python
Move.UP      # (row-1, col)
Move.DOWN    # (row+1, col)
Move.LEFT    # (row, col-1)
Move.RIGHT   # (row, col+1)
Move.STAY    # (row, col)
```

### Input Parameters to `step()`

- `map_state`: 2D numpy array (0=empty, 1=wall)
- `my_position`: (row, col) tuple
- `enemy_position`: (row, col) tuple
- `step_number`: int (starts at 1)

### Essential Helper Functions

```python
_is_valid_position(pos, map_state)  # Check if position is valid
_apply_move(pos, move)               # Apply move to position
_get_neighbors(pos, map_state)       # Get valid neighbors
_manhattan_distance(pos1, pos2)      # Calculate distance
```

### Common Algorithms

- **BFS**: Shortest path (optimal for Pacman)
- **A\***: Optimal with heuristic (efficient for Pacman)
- **Greedy**: Fast but not optimal
- **Minimax**: Adversarial search (good for Ghost)

---

## Checklist Before Submission

- [ ] Agent loads without errors
- [ ] Agent doesn't crash during game
- [ ] Agent makes valid moves (returns Move enum)
- [ ] Agent performs better than random
- [ ] Agent handles being trapped in corners
- [ ] Agent works for full game length (200 steps)
- [ ] Code is well-documented with comments
- [ ] No print statements in final version (or minimal)

---

## Getting Help

1. **Read this guide** thoroughly
2. **Check example_student/agent.py** for working code
3. **Test with --delay 0.5** to see what your agent is doing
4. **Use print statements** to debug
5. **Ask your instructor or TA** if stuck

---

## Good Luck! 🎮

Have fun implementing your AI agent! Remember:

- Start simple (get it working first)
- Test frequently
- Improve incrementally
- Learn from mistakes
- Compete fairly and have fun!

**May the best algorithm win!** 🏆

---

**Created**: October 2025  
**For**: AI Search Algorithms Course  
**Framework Version**: 1.0
3. **Goal Test**: 
   - Pacman: Reach Ghost's position
   - Ghost: Stay away from Pacman
4. **Path Cost**: Number of moves

### Suggested Algorithms

#### For Pacman (Seeker)

**Breadth-First Search (BFS)**
- Finds shortest path
- Guaranteed optimal
- Good starting point

```python
def bfs(self, start, goal, map_state):
    from collections import deque
    
    queue = deque([(start, [])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        if current == goal:
            return path
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(current, move)
            
            if next_pos not in visited and self._is_valid(next_pos, map_state):
                visited.add(next_pos)
                queue.append((next_pos, path + [move]))
    
    return [Move.STAY]
```

**A\* Search**
- Uses heuristic for efficiency
- Optimal with admissible heuristic
- Faster than BFS

```python
def astar(self, start, goal, map_state):
    from heapq import heappush, heappop
    
    def heuristic(pos):
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    frontier = [(0, start, [])]
    visited = set()
    
    while frontier:
        f_cost, current, path = heappop(frontier)
        
        if current == goal:
            return path
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(current, move)
            
            if self._is_valid(next_pos, map_state):
                new_path = path + [move]
                g_cost = len(new_path)
                f_cost = g_cost + heuristic(next_pos)
                heappush(frontier, (f_cost, next_pos, new_path))
    
    return [Move.STAY]
```

#### For Ghost (Hider)

**Maximize Distance**
- Find position furthest from Pacman
- Can use BFS or A*

```python
def find_furthest_position(self, my_pos, enemy_pos, map_state):
    from collections import deque
    
    # BFS from Pacman's position
    queue = deque([(enemy_pos, 0)])
    visited = {enemy_pos: 0}
    
    while queue:
        current, dist = queue.popleft()
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(current, move)
            
            if next_pos not in visited and self._is_valid(next_pos, map_state):
                visited[next_pos] = dist + 1
                queue.append((next_pos, dist + 1))
    
    # Find reachable position with maximum distance
    max_dist = -1
    best_positions = []
    
    for pos, dist in visited.items():
        if dist > max_dist:
            max_dist = dist
            best_positions = [pos]
        elif dist == max_dist:
            best_positions.append(pos)
    
    # Now find path to one of the best positions
    # ... (use BFS/A* to get there)
```

## Helper Functions

### Checking Valid Positions

```python
def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
    """Check if a position is valid (not a wall and within bounds)."""
    row, col = pos
    height, width = map_state.shape
    
    if row < 0 or row >= height or col < 0 or col >= width:
        return False
    
    return map_state[row, col] == 0
```

### Applying Moves

```python
def _apply_move(self, pos: tuple, move: Move) -> tuple:
    """Apply a move to a position."""
    delta_row, delta_col = move.value
    return (pos[0] + delta_row, pos[1] + delta_col)
```

### Getting Neighbors

```python
def _get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
    """Get all valid neighboring positions."""
    neighbors = []
    
    for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
        next_pos = self._apply_move(pos, move)
        if self._is_valid_position(next_pos, map_state):
            neighbors.append((next_pos, move))
    
    return neighbors
```

## Debugging Tips

1. **Print Statements**: They will appear in the terminal
```python
print(f"Step {step_number}: Moving {move.name}")
```

2. **Test Incrementally**: Start with simple logic, then improve

3. **Visualize**: Watch the game play out to understand behavior

4. **Handle Edge Cases**: 
   - What if no path exists?
   - What if trapped in a corner?
   - What if enemy is unreachable?

## Common Errors

### "Agent file not found"
- Make sure your folder is in `submissions/`
- Check that the file is named exactly `agent.py`

### "Must define a 'PacmanAgent' class"
- Your class name must be exactly `PacmanAgent` or `GhostAgent`
- Check for typos

### "Returned invalid move type"
- Always return a `Move` enum value
- Import: `from environment import Move`
- Return one of: `Move.UP`, `Move.DOWN`, `Move.LEFT`, `Move.RIGHT`, `Move.STAY`

### "AttributeError" or "NameError"
- Check your imports
- Make sure all helper methods are defined

## Testing Checklist

- [ ] Agent loads without errors
- [ ] Agent doesn't crash during game
- [ ] Agent makes valid moves
- [ ] Agent performs better than random
- [ ] Agent handles being trapped
- [ ] Agent works for full 200 steps

## Advanced Challenges

Once you have a working agent, try to improve it:

1. **Optimize Performance**: Make your search faster
2. **Better Heuristics**: For A* or greedy search
3. **Adversarial Thinking**: Predict opponent's moves
4. **Dynamic Strategy**: Change behavior based on game state
5. **Learning**: Analyze past games to improve

## Getting Help

- Read the README.md for detailed documentation
- Check the example_student implementation
- Review your algorithm pseudocode
- Test with `--no-viz` for faster iteration
- Use `--delay 0.5` for slower visualization

Good luck! 🎮
