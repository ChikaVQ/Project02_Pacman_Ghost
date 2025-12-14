"""
Main arena module that runs the game between two agents.
"""

import argparse
import math
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict

from environment import Environment, Move
from agent_loader import AgentLoader, AgentLoadError
from visualizer import GameVisualizer


class AgentTimeoutError(Exception):
    """Raised when an agent exceeds the allowed time per step."""


def _agent_timeout_handler(signum, frame):
    raise AgentTimeoutError("Agent step exceeded the allowed time")


def _start_alarm(seconds: float):
    if hasattr(signal, "setitimer"):
        signal.setitimer(signal.ITIMER_REAL, seconds)
    else:
        signal.alarm(max(1, int(math.ceil(seconds))))


def _cancel_alarm():
    if hasattr(signal, "setitimer"):
        signal.setitimer(signal.ITIMER_REAL, 0)
    else:
        signal.alarm(0)


class Arena:
    """
    Main arena class that orchestrates the game between agents.
    """
    
    def __init__(self, 
                 pacman_id: str, 
                 ghost_id: str,
                 submissions_dir: str = "../submissions",
                 max_steps: int = 200,
                 visualize: bool = True,
                 delay: float = 0.1,
                 step_timeout: Optional[float] = 3.0,
                 deterministic_starts: bool = True,
                 capture_distance_threshold: int = 1,
                 pacman_speed: int = 1,
                 pacman_obs_radius: int = 0,
                 ghost_obs_radius: int = 0):
        """
        Initialize the arena.
        
        Args:
            pacman_id: Student ID for Pacman agent
            ghost_id: Student ID for Ghost agent
            submissions_dir: Directory containing student submissions
            max_steps: Maximum number of steps before draw
            visualize: Whether to display the game
            delay: Delay between steps in seconds (for visualization)
            step_timeout: Max seconds allowed per agent step (>0 to enable)
            deterministic_starts: Use fixed classic start positions when available
            capture_distance_threshold: Manhattan distance under which Pacman captures
            pacman_speed: Maximum straight-path speed multiplier for Pacman
            pacman_obs_radius: Observation radius for Pacman (0 = full visibility)
            ghost_obs_radius: Observation radius for Ghost (0 = full visibility)
        """
        self.pacman_id = pacman_id
        self.ghost_id = ghost_id
        self.submissions_dir = submissions_dir
        self.max_steps = max_steps
        self.visualize = visualize
        self.delay = delay if visualize else 0.0
        self.step_timeout = step_timeout if step_timeout and step_timeout > 0 else None
        self.deterministic_starts = deterministic_starts
        self.capture_distance_threshold = max(1, int(capture_distance_threshold))
        self.pacman_speed = max(1, int(pacman_speed))
        self.pacman_obs_radius = max(0, int(pacman_obs_radius))
        self.ghost_obs_radius = max(0, int(ghost_obs_radius))
        self._timeout_supported = hasattr(signal, "SIGALRM")
        if self.step_timeout and not self._timeout_supported:
            print("WARNING: Step timeout requested but SIGALRM is unavailable on this platform. Timeout disabled.")
            self.step_timeout = None
        
        # Initialize components
        self.env = Environment(
            max_steps=max_steps,
            deterministic_starts=self.deterministic_starts,
            capture_distance_threshold=self.capture_distance_threshold,
            pacman_speed=self.pacman_speed
        )
        self.loader = AgentLoader(submissions_dir=submissions_dir)
        self.visualizer = GameVisualizer() if visualize else None
        
        # Load agents
        self.pacman_agent = None
        self.ghost_agent = None
        
        # Game statistics
        self.stats = {
            'total_steps': 0,
            'pacman_moves': [],
            'ghost_moves': [],
            'positions_history': []
        }
    
    def load_agents(self):
        """Load both agents from student submissions."""
        print(f"\n{'='*60}")
        print(f"{'ARENA INITIALIZATION':^60}")
        print(f"{'='*60}\n")
        
        try:
            print(f"Loading Pacman agent from student: {self.pacman_id}")
            self.pacman_agent = self.loader.load_agent(
                self.pacman_id,
                'pacman',
                init_kwargs={'pacman_speed': self.pacman_speed}
            )
            print(f"✓ Pacman agent loaded successfully\n")
        except AgentLoadError as e:
            print(f"✗ Failed to load Pacman agent: {e}\n")
            sys.exit(1)
        
        try:
            print(f"Loading Ghost agent from student: {self.ghost_id}")
            self.ghost_agent = self.loader.load_agent(self.ghost_id, 'ghost')
            print(f"✓ Ghost agent loaded successfully\n")
        except AgentLoadError as e:
            print(f"✗ Failed to load Ghost agent: {e}\n")
            sys.exit(1)
    
    def run_game(self) -> Tuple[str, Dict]:
        """
        Run the game until completion.
        
        Returns:
            Tuple of (result, statistics)
            - result: 'pacman_wins', 'ghost_wins', or 'draw'
            - statistics: Dictionary containing game statistics
        """
        # Reset environment
        map_state, pacman_pos, ghost_pos = self.env.reset()
        
        print(f"{'='*60}")
        print(f"{'GAME START':^60}")
        print(f"{'='*60}\n")
        print(f"Pacman (Seeker): {self.pacman_id} at position {pacman_pos}")
        print(f"Ghost (Hider): {self.ghost_id} at position {ghost_pos}")
        print(f"Maximum steps: {self.max_steps}\n")
        
        if self.visualize:
            self.visualizer.display(self.env, 0, self.pacman_id, self.ghost_id)
        
        game_over = False
        result = ''
        step = 0
        
        while not game_over:
            step += 1
            
            # Get observations for each agent
            pacman_obs, pacman_my_pos, pacman_visible_enemy = self.env.get_observation(
                'pacman', self.pacman_obs_radius, self.ghost_obs_radius
            )
            ghost_obs, ghost_my_pos, ghost_visible_enemy = self.env.get_observation(
                'ghost', self.pacman_obs_radius, self.ghost_obs_radius
            )
            
            # Get moves from both agents
            try:
                pacman_action = self._run_agent_step(
                    lambda: self.pacman_agent.step(
                        pacman_obs, pacman_my_pos, pacman_visible_enemy, step
                    )
                )
                pacman_action = self.loader.validate_agent_move(
                    pacman_action,
                    'pacman',
                    self.pacman_id,
                    self.pacman_speed
                )
            except AgentTimeoutError:
                print(f"\n✗ Pacman agent timed out at step {step} after {self.step_timeout}s")
                print("Ghost wins by default!")
                result = 'ghost_wins'
                game_over = True
                break
            except Exception as e:
                print(f"\n✗ Error in Pacman agent at step {step}: {e}")
                print(f"Ghost wins by default!")
                result = 'ghost_wins'
                game_over = True
                break
            
            try:
                ghost_move = self._run_agent_step(
                    lambda: self.ghost_agent.step(
                        ghost_obs, ghost_my_pos, ghost_visible_enemy, step
                    )
                )
                ghost_move = self.loader.validate_agent_move(
                    ghost_move,
                    'ghost',
                    self.ghost_id
                )
            except AgentTimeoutError:
                print(f"\n✗ Ghost agent timed out at step {step} after {self.step_timeout}s")
                print(f"Pacman wins by default!")
                result = 'pacman_wins'
                game_over = True
                break
            except Exception as e:
                print(f"\n✗ Error in Ghost agent at step {step}: {e}")
                print(f"Pacman wins by default!")
                result = 'pacman_wins'
                game_over = True
                break
            
            # Record moves
            self.stats['pacman_moves'].append(pacman_action)
            self.stats['ghost_moves'].append(ghost_move)
            
            # Execute step in environment
            game_over, result, new_state = self.env.step(pacman_action, ghost_move)
            map_state, pacman_pos, ghost_pos = new_state
            
            # Record position history
            self.stats['positions_history'].append((pacman_pos, ghost_pos))
            
            # Visualize if enabled
            if self.visualize:
                time.sleep(self.delay)
                self.visualizer.display(
                    self.env, step, self.pacman_id, self.ghost_id,
                    pacman_action, ghost_move, result if game_over else None
                )
        
        self.stats['total_steps'] = step
        
        # Display final results
        self.display_results(result)
        
        return result, self.stats
    
    def display_results(self, result: str):
        """
        Display the final game results.
        
        Args:
            result: Game result ('pacman_wins', 'ghost_wins', or 'draw')
        """
        print(f"\n{'='*60}")
        print(f"{'GAME OVER':^60}")
        print(f"{'='*60}\n")
        
        if result == 'pacman_wins':
            print(f"🏆 WINNER: {self.pacman_id} (Pacman)")
            print(f"   Pacman caught the Ghost!")
        elif result == 'ghost_wins':
            print(f"🏆 WINNER: {self.ghost_id} (Ghost)")
            print(f"   Ghost successfully evaded Pacman!")
        elif result == 'draw':
            print(f"🤝 DRAW")
            print(f"   Maximum steps ({self.max_steps}) reached without capture")
        
        print(f"\nGame Statistics:")
        print(f"  Total Steps: {self.stats['total_steps']}")
        print(f"  Final Distance: {self.env.get_distance(self.env.pacman_pos, self.env.ghost_pos)}")
        print(f"\n{'='*60}\n")

    def _run_agent_step(self, step_callable):
        if not self.step_timeout or self.step_timeout <= 0:
            return step_callable()

        previous_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _agent_timeout_handler)
            _start_alarm(self.step_timeout)
            return step_callable()
        finally:
            _cancel_alarm()
            signal.signal(signal.SIGALRM, previous_handler)


def main():
    """Main entry point for the arena."""
    parser = argparse.ArgumentParser(
        description="Pacman vs Ghost Arena - AI Search Algorithms Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python arena.py --seek student1 --hide student2
  python arena.py --seek student1 --hide student2 --max-steps 300 --no-viz
  python arena.py --seek alice --hide bob --delay 0.5
        """
    )
    
    parser.add_argument(
        '--seek', '--pacman',
        dest='seek',
        required=True,
        help='Student ID for the Pacman (seeker) agent'
    )
    
    parser.add_argument(
        '--hide', '--ghost',
        dest='hide',
        required=True,
        help='Student ID for the Ghost (hider) agent'
    )
    
    parser.add_argument(
        '--submissions-dir',
        default='../submissions',
        help='Directory containing student submissions (default: ../submissions)'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=200,
        help='Maximum number of steps before draw (default: 200)'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=0.1,
        help='Delay between steps in seconds (default: 0.1)'
    )

    parser.add_argument(
        '--step-timeout',
        type=float,
        default=1.0,
        help='Maximum seconds allowed per agent step (<=0 disables timeout)'
    )

    parser.add_argument(
        '--start-mode',
        choices=['deterministic', 'stochastic'],
        default='deterministic',
        help='Select deterministic classic starts or stochastic random starts'
    )

    parser.add_argument(
        '--capture-distance',
        type=int,
        default=1,
        help='Pacman captures Ghost when Manhattan distance is below this value'
    )

    parser.add_argument(
        '--pacman-speed',
        type=int,
        default=1,
        help='Maximum tiles Pacman can advance in the same direction when on a straight path'
    )

    parser.add_argument(
        '--pacman-obs-radius',
        type=int,
        default=0,
        help='Observation radius for Pacman; 0 = full visibility (default: 0)'
    )

    parser.add_argument(
        '--ghost-obs-radius',
        type=int,
        default=0,
        help='Observation radius for Ghost; 0 = full visibility (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Create and run arena
    arena = Arena(
        pacman_id=args.seek,
        ghost_id=args.hide,
        submissions_dir=args.submissions_dir,
        max_steps=args.max_steps,
        visualize=not args.no_viz,
        delay=args.delay,
        step_timeout=args.step_timeout,
        deterministic_starts=(args.start_mode == 'deterministic'),
        capture_distance_threshold=args.capture_distance,
        pacman_speed=args.pacman_speed,
        pacman_obs_radius=args.pacman_obs_radius,
        ghost_obs_radius=args.ghost_obs_radius
    )
    
    arena.load_agents()
    result, stats = arena.run_game()
    
    return 0 if result in ['pacman_wins', 'ghost_wins', 'draw'] else 1


if __name__ == '__main__':
    sys.exit(main())
