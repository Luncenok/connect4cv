# %% [markdown]
# ## Step 5: Game State Management Module
# This module handles game state tracking, winning condition detection, and event logging.

# %% [markdown]
# ### Game Grid Management

# %%
import numpy as np
from enum import Enum
import json
from datetime import datetime
from typing import Tuple, List, Optional

class CellState(Enum):
    EMPTY = 0
    RED = 1
    YELLOW = 2

class GameState:
    def __init__(self, rows=6, cols=7):
        """Initialize game state."""
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)
        self.counter_frames = np.zeros((rows, cols), dtype=int)  # Track frames counter is detected
        self.permanent_counters = np.zeros((rows, cols), dtype=bool)  # Track permanent counters
        self.game_started = False
        self.game_ended = False
        self.winner = None
        self.events = []
        
    def is_valid_move(self, col: int) -> bool:
        """Check if a move is valid."""
        return 0 <= col < self.cols and self.grid[0][col] == CellState.EMPTY.value

    def track_counter(self, row: int, col: int, state: int) -> bool:
        """Track counter detection and return if it became permanent."""
        if self.permanent_counters[row][col]:
            return True
            
        if self.grid[row][col] == state:
            self.counter_frames[row][col] += 1
        else:
            self.counter_frames[row][col] = 1
            self.grid[row][col] = state
            
        if self.counter_frames[row][col] >= 50:
            self.permanent_counters[row][col] = True
            return True
        return False
        
    def is_permanent(self, row: int, col: int) -> bool:
        """Check if counter at given position is permanent."""
        return self.permanent_counters[row][col]

# %% [markdown]
# ### Winning Condition Detection

# %%
class WinDetector:
    @staticmethod
    def check_win(grid: np.ndarray) -> Tuple[bool, List[Tuple[int, int]]]:
        """Check for winning conditions in all directions."""
        rows, cols = grid.shape
        
        # Check horizontal
        for row in range(rows):
            for col in range(cols-3):
                if grid[row][col] != CellState.EMPTY.value:
                    if all(grid[row][col] == grid[row][col+i] for i in range(4)):
                        return True, [(row, col+i) for i in range(4)]
        
        # Check vertical
        for row in range(rows-3):
            for col in range(cols):
                if grid[row][col] != CellState.EMPTY.value:
                    if all(grid[row+i][col] == grid[row][col] for i in range(4)):
                        return True, [(row+i, col) for i in range(4)]
        
        # Check diagonal (positive slope)
        for row in range(rows-3):
            for col in range(cols-3):
                if grid[row][col] != CellState.EMPTY.value:
                    if all(grid[row+i][col+i] == grid[row][col] for i in range(4)):
                        return True, [(row+i, col+i) for i in range(4)]
        
        # Check diagonal (negative slope)
        for row in range(3, rows):
            for col in range(cols-3):
                if grid[row][col] != CellState.EMPTY.value:
                    if all(grid[row-i][col+i] == grid[row][col] for i in range(4)):
                        return True, [(row-i, col+i) for i in range(4)]
        
        return False, []

# %% [markdown]
# ### Event Logging

# %%
class GameEvent:
    def __init__(self, event_type: str, details: dict, frame_number: int = 0):
        """Initialize game event."""
        self.event_type = event_type
        self.details = details
        self.timestamp = frame_number / 30.0  # Convert frame number to seconds (30 fps)
        
    def to_dict(self):
        """Convert event to dictionary."""
        return {
            'type': self.event_type,
            'details': self.details,
            'timestamp': self.timestamp
        }

class GameLogger:
    def __init__(self, output_file: str):
        """Initialize game logger."""
        self.output_file = output_file
        self.events = []
        
    def log_event(self, event_type: str, details: dict, frame_number: int = 0):
        """Log a game event."""
        event = GameEvent(event_type, details, frame_number)
        print(f"Event logged: {event.to_dict()}")
        self.events.append(event)
        self.save_log()
        
    def save_log(self):
        """Save event log to file."""
        with open(self.output_file, 'w') as f:
            json.dump([event.to_dict() for event in self.events], f, indent=2)

# %% [markdown]
# ### Game Controller

# %%
class GameController:
    def __init__(self, game_state: GameState, output_file: str = 'output/game_log.json'):
        """Initialize game controller."""
        self.game_state = game_state
        self.win_detector = WinDetector()
        self.logger = GameLogger(output_file)
        self.frame_number = 0
        self.winning_cells = None
        
        # Dice tracking
        self.dice_history = []  # List of last 10 dice values
        self.permanent_dice = 0  # Current permanent dice value
        self.permanent_dice_count = 0  # How many times permanent dice was seen in last 10 frames
        
        # Bench tracking
        self.bench_history = []  # List of last 60 frames of counter positions
        self.permanent_yellow_centers = []  # Current permanent yellow counter positions
        self.permanent_red_centers = []  # Current permanent red counter positions
        
    def process_move(self, row: int, col: int, player: CellState) -> Tuple[bool, Optional[List[Tuple[int, int]]]]:
        """Process a player's move. Returns (success, winning_cells)."""
        if not self.game_state.game_started:
            self.game_state.game_started = True
            self.logger.log_event('game_start', {}, self.frame_number)
            
        # Make the move
        self.game_state.grid[row][col] = player.value
        self.logger.log_event('move', {
            'player': player.name,
            'row': row,
            'col': col
        }, self.frame_number)

        self.logger.save_log()
        
        # Create a grid with only permanent counters for win checking
        permanent_grid = np.where(self.game_state.permanent_counters, self.game_state.grid, CellState.EMPTY.value)
        
        # Check for win using only permanent counters
        won, winning_cells = self.win_detector.check_win(permanent_grid)
        if won:
            self.game_state.game_ended = True
            self.game_state.winner = player
            self.winning_cells = winning_cells
            self.logger.log_event('game_end', {
                'winner': player.name,
                'winning_cells': winning_cells
            }, self.frame_number)
            self.logger.save_log()
            return True, winning_cells
            
        # Check for draw - only considering permanent counters
        playable_cells = permanent_grid[self.game_state.permanent_counters]
        if len(playable_cells) == self.game_state.rows*self.game_state.cols:
            self.game_state.game_ended = True
            self.logger.log_event('game_end', {'winner': None}, self.frame_number)
            self.logger.save_log()
            return True, None
            
        self.frame_number += 1
        return True, None

    def process_dice(self, dice_value: int) -> int:
        """Process detected dice value and return permanent value if valid."""
        history_len = 30
        min_consecutive = 10

        # Handle None or invalid values
        if dice_value is None:
            dice_value = 0
            
        # Add new value to history
        self.dice_history.append(dice_value)
        if len(self.dice_history) > history_len:
            self.dice_history.pop(0)
            
        # Count occurrences of each value in history
        value_counts = {}
        for value in self.dice_history:
            if value > 0:  # Only count valid dice values
                value_counts[value] = value_counts.get(value, 0) + 1
        
        # Find most frequent value
        max_count = 0
        max_value = 0
        for value, count in value_counts.items():
            if count > max_count:
                max_count = count
                max_value = value
        
        # Update permanent dice if necessary
        if max_count >= min_consecutive:  # Value seen in at least 5 out of 10 frames
            if not self.game_state.game_started:
                self.game_state.game_started = True
                self.logger.log_event('game_start', {}, self.frame_number)
            if max_value != self.permanent_dice:  # New permanent value
                self.permanent_dice = max_value
                self.logger.log_event('dice_detected', {'value': max_value}, self.frame_number)
            self.permanent_dice_count = max_count
        elif len(self.dice_history) >= history_len:  # Clear permanent dice if no value is frequent enough
            if not self.game_state.game_started:
                self.game_state.game_started = True
                self.logger.log_event('game_start', {}, self.frame_number)
            if self.permanent_dice != 0:
                self.logger.log_event('dice_cleared', {'value': self.permanent_dice}, self.frame_number)
            self.permanent_dice = 0
            self.permanent_dice_count = 0
        
        return self.permanent_dice if self.permanent_dice_count >= min_consecutive else 0

    def process_bench(self, yellow_centers: list, red_centers: list) -> Tuple[list, list]:
        """Process detected bench counters and return permanent positions if valid."""
        history_len = 120
        min_consecutive = 30
        position_tolerance = 20  # pixels
        
        # Add new centers to history
        self.bench_history.append({
            'yellow': yellow_centers,
            'red': red_centers
        })
        if len(self.bench_history) > history_len:
            self.bench_history.pop(0)
            
        def find_clusters(history_positions: list) -> list:
            if not history_positions:
                return []
            
            # Flatten all positions from history into a single list
            all_positions = []
            for positions in history_positions:
                all_positions.extend(positions)
                
            if not all_positions:
                return []
            
            # Find clusters
            clusters = []
            used = set()
            
            for i, pos1 in enumerate(all_positions):
                if i in used:
                    continue
                    
                # Start new cluster
                cluster = [pos1]
                used.add(i)
                
                # Find all positions close to this one
                for j, pos2 in enumerate(all_positions):
                    if j in used:
                        continue
                    dx = pos1[0] - pos2[0]
                    dy = pos1[1] - pos2[1]
                    if (dx * dx + dy * dy) ** 0.5 < position_tolerance:
                        cluster.append(pos2)
                        used.add(j)
                
                # Calculate cluster center
                if len(cluster) > min_consecutive: 
                    center_x = sum(p[0] for p in cluster) // len(cluster)
                    center_y = sum(p[1] for p in cluster) // len(cluster)
                    clusters.append((center_x, center_y))
            
            return clusters
        
        # Get all positions from history
        yellow_history = [frame.get('yellow', []) for frame in self.bench_history]
        red_history = [frame.get('red', []) for frame in self.bench_history]
        
        # Find cluster centers
        new_yellow = find_clusters(yellow_history)
        new_red = find_clusters(red_history)
        
        # Update permanent positions if they've changed
        if len(set(map(tuple, self.permanent_yellow_centers))) != len(set(map(tuple, new_yellow))):
            self.permanent_yellow_centers = new_yellow
            self.logger.log_event('bench_yellow_update', 
                                {'count': len(new_yellow)}, 
                                self.frame_number)
            
        if len(set(map(tuple, self.permanent_red_centers))) != len(set(map(tuple, new_red))):
            self.permanent_red_centers = new_red
            self.logger.log_event('bench_red_update', 
                                {'count': len(new_red)}, 
                                self.frame_number)
            
        return self.permanent_yellow_centers, self.permanent_red_centers
