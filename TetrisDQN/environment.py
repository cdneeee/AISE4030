"""
environment.py – Custom Tetris Gymnasium Environment
=====================================================
Implements a standard 20 × 10 Tetris board as a Gymnasium-compatible
environment.  This module is the sole source of environment logic; all
other modules interact with Tetris only through this API.

State Space (Box, float32, shape=(220,)):
    - Board occupancy  : 200 values  in {0, 1}   (20 × 10 grid, row-major)
    - Current piece    :   7 values  one-hot      (piece type 0–6)
    - Orientation      :   4 values  one-hot      (clockwise rotation 0–3)
    - Column (norm.)   :   1 value   in [0, 1]    (leftmost col of bounding box)
    - Row    (norm.)   :   1 value   in [0, 1]    (topmost row of bounding box)
    - Next piece       :   7 values  one-hot      (preview piece type)
    Total              : 220 float32 values in [0, 1]

Action Space (Discrete(6)):
    0  LEFT     – shift piece one column left
    1  RIGHT    – shift piece one column right
    2  ROT_CW   – rotate 90° clockwise
    3  ROT_CCW  – rotate 90° counter-clockwise
    4  DROP     – hard-drop (instant fall to lowest valid row, then lock)
    5  NO_OP    – no lateral/rotation action (gravity still applies)

Reward:
    +LINE_CLEAR_SCORES[n]  on piece lock  (n ∈ {0,1,2,3,4} lines cleared)
    −0.5 × Δholes          on piece lock  (penalty for new holes created)
    −0.3 × Δheight         on piece lock  (penalty for aggregate height increase)
    −0.01                  every step     (survival time penalty)
    −100                   on game over

Episode Termination:
    Triggered when a newly spawned piece immediately overlaps occupied cells.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --------------------------------------------------------------------------- #
#  Tetromino definitions                                                        #
# --------------------------------------------------------------------------- #
# Each entry is the base (rotation-0) shape as a 2-D binary NumPy array.
# Clockwise rotations are derived on-the-fly via np.rot90.

BASE_SHAPES = [
    np.array([[1, 1, 1, 1]]),             # 0: I
    np.array([[1, 1], [1, 1]]),           # 1: O
    np.array([[0, 1, 0], [1, 1, 1]]),     # 2: T
    np.array([[0, 1, 1], [1, 1, 0]]),     # 3: S
    np.array([[1, 1, 0], [0, 1, 1]]),     # 4: Z
    np.array([[1, 0, 0], [1, 1, 1]]),     # 5: J
    np.array([[0, 0, 1], [1, 1, 1]]),     # 6: L
]

PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]

# Tetris scoring table: number of lines cleared -> score gained
LINE_CLEAR_SCORES = [0, 100, 300, 500, 800]


# --------------------------------------------------------------------------- #
#  Environment class                                                            #
# --------------------------------------------------------------------------- #

class TetrisEnv(gym.Env):
    """
    Custom Tetris environment conforming to the Gymnasium API.

    The agent controls a falling tetromino on a 20 × 10 board.  At each
    time-step the agent picks one of six discrete actions; gravity is then
    applied (piece drops one row).  When a piece can no longer fall it is
    locked in place, complete rows are cleared, and the next piece spawns.

    The MDP formulation follows the Phase 1 proposal:
        s_t = (B, p_t, o_t, x_t, y_t, p_{t+1})
    where B is the binary board, p_t the current piece type, o_t its
    orientation, (x_t, y_t) the bounding-box position, and p_{t+1} the
    preview piece.

    Args:
        config (dict, optional): Runtime overrides.  Supported keys:
            'rows' (int) – board height (default 20).
            'cols' (int) – board width  (default 10).
        render_mode (str, optional): 'human' or 'rgb_array'.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # ------------------------------------------------------------------ #
    #  Class-level constants                                               #
    # ------------------------------------------------------------------ #
    ROWS = 20
    COLS = 10
    NUM_PIECES = 7
    NUM_ACTIONS = 6

    # Action index constants
    LEFT    = 0
    RIGHT   = 1
    ROT_CW  = 2
    ROT_CCW = 3
    DROP    = 4
    NO_OP   = 5

    def __init__(self, config: dict = None, render_mode: str = None):
        """
        Initialise spaces, board dimensions, and internal state variables.

        Args:
            config (dict, optional): Configuration overrides.
                Supported keys: 'rows' (int), 'cols' (int).
            render_mode (str, optional): Rendering backend.
                One of 'human', 'rgb_array', or None (headless).
        """
        super().__init__()
        self.render_mode = render_mode

        cfg = config or {}
        self.ROWS = int(cfg.get("rows", self.ROWS))
        self.COLS = int(cfg.get("cols", self.COLS))

        # Observation dimension:
        #   board(R*C) + piece(7) + rotation(4) + col(1) + row(1) + next(7)
        self.obs_dim = self.ROWS * self.COLS + 7 + 4 + 1 + 1 + 7

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)

        # Internal state – populated on reset()
        self.board:            np.ndarray = None
        self.current_piece:    int        = None
        self.current_rotation: int        = None
        self.current_row:      int        = None
        self.current_col:      int        = None
        self.next_piece:       int        = None
        self.score:            int        = 0
        self.lines_cleared:    int        = 0
        self.steps:            int        = 0

        # Pygame handles (lazy-initialised on first render call)
        self._screen = None
        self._clock  = None

    # ------------------------------------------------------------------ #
    #  Gymnasium core API                                                  #
    # ------------------------------------------------------------------ #

    def reset(self, seed: int = None, options: dict = None):
        """
        Reset the environment to a fresh game state.

        Clears the board, zeroes all counters, draws two random pieces (the
        active piece and the preview), and places the active piece at the
        top-centre of the board.

        Args:
            seed (int, optional): RNG seed for reproducibility.
            options (dict, optional): Unused; reserved for future extensions.

        Returns:
            tuple:
                observation (np.ndarray): Initial state vector, shape (220,).
                info (dict): Empty dict (no extra info at reset).
        """
        super().reset(seed=seed)

        self.board         = np.zeros((self.ROWS, self.COLS), dtype=np.int32)
        self.score         = 0
        self.lines_cleared = 0
        self.steps         = 0

        # Draw first preview piece, then spawn the active piece
        self.next_piece = int(self.np_random.integers(0, self.NUM_PIECES))
        self._spawn_piece()

        return self._get_observation(), {}

    def step(self, action: int):
        """
        Apply one action and advance the simulation by one time-step.

        Order of events per step:
            1. Apply the chosen lateral / rotation action (or hard-drop).
            2. For non-DROP actions: apply gravity (piece falls one row).
            3. If the piece cannot fall further: lock it, clear lines, spawn next.
            4. If spawning the next piece is blocked: game over.

        Args:
            action (int): Discrete action index in [0, 5].

        Returns:
            tuple:
                observation (np.ndarray): New state vector, shape (220,).
                reward (float): Immediate scalar reward.
                terminated (bool): True if the episode has ended (game over).
                truncated (bool): Always False (no external step limit).
                info (dict): Diagnostic values:
                    'score' (int): Cumulative in-game score.
                    'lines_cleared' (int): Total lines cleared this episode.
                    'steps' (int): Total steps taken this episode.
        """
        self.steps += 1
        reward     = -0.01  # per-step time penalty
        terminated = False

        prev_holes  = self._count_holes()
        prev_height = self._aggregate_height()

        if action == self.DROP:
            # Hard drop: fall to the lowest valid row, then lock immediately
            while self._is_valid(
                self.current_piece, self.current_rotation,
                self.current_row + 1, self.current_col
            ):
                self.current_row += 1
            reward += self._lock_and_clear(prev_holes, prev_height)
            if not self._spawn_piece():
                terminated = True
                reward -= 100.0
        else:
            # 1. Lateral / rotation action
            self._apply_action(action)

            # 2. Gravity
            if self._is_valid(
                self.current_piece, self.current_rotation,
                self.current_row + 1, self.current_col
            ):
                self.current_row += 1
            else:
                # 3. Lock, clear, spawn
                reward += self._lock_and_clear(prev_holes, prev_height)
                if not self._spawn_piece():
                    terminated = True
                    reward -= 100.0

        info = {
            "score":         self.score,
            "lines_cleared": self.lines_cleared,
            "steps":         self.steps,
        }
        return self._get_observation(), float(reward), terminated, False, info

    def render(self):
        """
        Render the current board state via Pygame.

        In 'human' mode: opens a window and displays the board live.
        In 'rgb_array' mode: returns an (H, W, 3) uint8 pixel array.

        Returns:
            np.ndarray or None:
                RGB pixel array of shape (ROWS*cell_size, COLS*cell_size, 3)
                when render_mode is 'rgb_array'; None otherwise.

        Raises:
            ImportError: If pygame is not installed.
        """
        if self.render_mode is None:
            return None

        try:
            import pygame
        except ImportError:
            raise ImportError("Install pygame to use render(): pip install pygame")

        cell_size = 30
        width  = self.COLS * cell_size
        height = self.ROWS * cell_size

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption("Tetris RL Agent")
            else:
                self._screen = pygame.Surface((width, height))
            self._clock = pygame.time.Clock()

        self._screen.fill((10, 10, 10))

        # Draw locked board cells
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.board[r, c]:
                    rect = pygame.Rect(
                        c * cell_size, r * cell_size,
                        cell_size - 1, cell_size - 1
                    )
                    pygame.draw.rect(self._screen, (80, 120, 200), rect)

        # Draw the active (falling) piece in a different colour
        for r, c in self._get_cells(
            self.current_piece, self.current_rotation,
            self.current_row, self.current_col
        ):
            if 0 <= r < self.ROWS and 0 <= c < self.COLS:
                rect = pygame.Rect(
                    c * cell_size, r * cell_size,
                    cell_size - 1, cell_size - 1
                )
                pygame.draw.rect(self._screen, (220, 200, 50), rect)

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)),
                axes=(1, 0, 2),
            )

    def close(self):
        """
        Tear down the Pygame rendering window if it was opened.

        Safe to call even if render() was never invoked.
        """
        if self._screen is not None:
            try:
                import pygame
                pygame.quit()
            except ImportError:
                pass
            self._screen = None

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_shape(self, piece_idx: int, rotation: int) -> np.ndarray:
        """
        Return the 2-D binary array for a piece at a given clockwise rotation.

        Uses np.rot90 with k=-rotation to achieve clockwise rotation.

        Args:
            piece_idx (int): Piece index in [0, 6].
            rotation (int): Clockwise rotation count in [0, 3].

        Returns:
            np.ndarray: 2-D binary array (dtype int) representing the shape.
        """
        return np.rot90(BASE_SHAPES[piece_idx], k=-rotation)

    def _get_cells(
        self, piece_idx: int, rotation: int, row: int, col: int
    ) -> list:
        """
        Return the absolute (row, col) board coordinates of all filled cells.

        The piece's bounding box top-left corner is at (row, col).  Cells
        with row < 0 are above the board (valid during spawn / early fall).

        Args:
            piece_idx (int): Piece index in [0, 6].
            rotation (int): Clockwise rotation count in [0, 3].
            row (int): Bounding-box top row (may be negative).
            col (int): Bounding-box left column.

        Returns:
            list[tuple[int, int]]: Board (row, col) for each filled cell.
        """
        shape = self._get_shape(piece_idx, rotation)
        cells = []
        for r in range(shape.shape[0]):
            for c in range(shape.shape[1]):
                if shape[r, c]:
                    cells.append((row + r, col + c))
        return cells

    def _is_valid(
        self, piece_idx: int, rotation: int, row: int, col: int
    ) -> bool:
        """
        Check whether a piece placement is collision-free and within bounds.

        Cells above the board (row < 0) are allowed (piece spawning).
        Cells below the board or outside the columns are invalid.

        Args:
            piece_idx (int): Piece index in [0, 6].
            rotation (int): Clockwise rotation count in [0, 3].
            row (int): Proposed bounding-box top row.
            col (int): Proposed bounding-box left column.

        Returns:
            bool: True if the placement is valid.
        """
        for r, c in self._get_cells(piece_idx, rotation, row, col):
            if c < 0 or c >= self.COLS:
                return False
            if r >= self.ROWS:
                return False
            if r >= 0 and self.board[r, c]:
                return False
        return True

    def _apply_action(self, action: int):
        """
        Apply a lateral shift or rotation to the active piece (no gravity).

        Invalid moves (would cause collision) are silently ignored.
        DROP and NO_OP are handled in step(); only LEFT/RIGHT/ROT_CW/ROT_CCW
        are processed here.

        Args:
            action (int): Action index.  One of LEFT, RIGHT, ROT_CW, ROT_CCW,
                or NO_OP (0–3, 5).
        """
        if action == self.LEFT:
            if self._is_valid(
                self.current_piece, self.current_rotation,
                self.current_row, self.current_col - 1
            ):
                self.current_col -= 1

        elif action == self.RIGHT:
            if self._is_valid(
                self.current_piece, self.current_rotation,
                self.current_row, self.current_col + 1
            ):
                self.current_col += 1

        elif action == self.ROT_CW:
            new_rot = (self.current_rotation + 1) % 4
            if self._is_valid(
                self.current_piece, new_rot,
                self.current_row, self.current_col
            ):
                self.current_rotation = new_rot

        elif action == self.ROT_CCW:
            new_rot = (self.current_rotation - 1) % 4
            if self._is_valid(
                self.current_piece, new_rot,
                self.current_row, self.current_col
            ):
                self.current_rotation = new_rot
        # NO_OP: nothing to do

    def _lock_piece(self):
        """
        Write the active piece's cells permanently onto the board array.

        Only cells within the valid row range [0, ROWS) are written;
        cells above the board (row < 0) are discarded.
        """
        for r, c in self._get_cells(
            self.current_piece, self.current_rotation,
            self.current_row, self.current_col
        ):
            if 0 <= r < self.ROWS and 0 <= c < self.COLS:
                self.board[r, c] = 1

    def _clear_lines(self) -> int:
        """
        Remove all fully-occupied rows and drop remaining rows downward.

        Empty replacement rows are inserted at the top of the board.

        Returns:
            int: Number of lines cleared (0 to 4 inclusive).
        """
        full_rows = np.where(np.all(self.board, axis=1))[0]
        n = len(full_rows)
        if n > 0:
            keep = np.setdiff1d(np.arange(self.ROWS), full_rows)
            new_board = np.zeros((self.ROWS, self.COLS), dtype=np.int32)
            new_board[self.ROWS - len(keep):] = self.board[keep]
            self.board = new_board
        return n

    def _lock_and_clear(self, prev_holes: int, prev_height: int) -> float:
        """
        Lock the active piece, clear lines, and compute the placement reward.

        Combines _lock_piece() and _clear_lines() so that reward shaping
        (hole and height penalties) is computed relative to the board state
        immediately before the piece was placed.

        Args:
            prev_holes (int): Total hole count before this piece was placed.
            prev_height (int): Aggregate column height before placement.

        Returns:
            float: Reward contribution from this placement event.
        """
        self._lock_piece()
        n_lines = self._clear_lines()
        self.lines_cleared += n_lines
        score_gain = LINE_CLEAR_SCORES[n_lines]
        self.score += score_gain

        new_holes  = self._count_holes()
        new_height = self._aggregate_height()

        reward  = float(score_gain)
        reward -= 0.5 * max(0, new_holes  - prev_holes)
        reward -= 0.3 * max(0, new_height - prev_height)
        return reward

    def _spawn_piece(self) -> bool:
        """
        Promote the preview piece to active, draw a new preview, and position.

        The piece spawns at rotation 0 and is centred horizontally at the
        top of the board (row 0).  Returns False (game over) when the spawn
        position is already occupied.

        Returns:
            bool: True if the spawn is valid and the game continues;
                  False if the board is full and the episode must end.
        """
        self.current_piece    = self.next_piece
        self.current_rotation = 0
        self.next_piece       = int(self.np_random.integers(0, self.NUM_PIECES))

        shape = self._get_shape(self.current_piece, 0)
        self.current_col = (self.COLS - shape.shape[1]) // 2
        self.current_row = 0

        return self._is_valid(
            self.current_piece, self.current_rotation,
            self.current_row, self.current_col,
        )

    def _count_holes(self) -> int:
        """
        Count empty cells that have at least one filled cell directly above.

        A hole is any empty cell (board == 0) in a column below the topmost
        filled cell of that column.  Holes reduce future line-clear potential.

        Returns:
            int: Total number of holes across all columns.
        """
        holes = 0
        for c in range(self.COLS):
            col = self.board[:, c]
            first_filled = int(np.argmax(col))
            if col[first_filled]:  # column has at least one filled cell
                holes += int(np.sum(col[first_filled:] == 0))
        return holes

    def _aggregate_height(self) -> int:
        """
        Compute the aggregate height: sum of the height of each column.

        Column height is defined as ROWS minus the row index of the topmost
        filled cell.  Empty columns contribute 0.

        Returns:
            int: Sum of all column heights.
        """
        total = 0
        for c in range(self.COLS):
            col = self.board[:, c]
            first_filled = int(np.argmax(col))
            if col[first_filled]:
                total += self.ROWS - first_filled
        return total

    def _get_observation(self) -> np.ndarray:
        """
        Construct and return the flat observation vector for the current state.

        Encodes:  board (binary flat) | current piece (one-hot) |
                  rotation (one-hot) | col (normalised) | row (normalised) |
                  next piece (one-hot).

        Returns:
            np.ndarray: Float32 array of shape (obs_dim,) with values in [0, 1].
        """
        board_flat = self.board.flatten().astype(np.float32)

        piece_oh = np.zeros(self.NUM_PIECES, dtype=np.float32)
        piece_oh[self.current_piece] = 1.0

        rot_oh = np.zeros(4, dtype=np.float32)
        rot_oh[self.current_rotation] = 1.0

        col_norm = np.array(
            [self.current_col / max(1, self.COLS - 1)], dtype=np.float32
        )
        row_norm = np.array(
            [max(0, self.current_row) / max(1, self.ROWS - 1)], dtype=np.float32
        )

        next_oh = np.zeros(self.NUM_PIECES, dtype=np.float32)
        next_oh[self.next_piece] = 1.0

        return np.concatenate(
            [board_flat, piece_oh, rot_oh, col_norm, row_norm, next_oh]
        )


# --------------------------------------------------------------------------- #
#  Factory function                                                             #
# --------------------------------------------------------------------------- #

def make_env(config: dict = None, render_mode: str = None) -> TetrisEnv:
    """
    Construct and return a configured TetrisEnv instance.

    This factory is the recommended way for other modules to obtain an
    environment; it allows future wrappers (e.g. frame-stacking, reward
    normalisation) to be added here without changing caller code.

    Args:
        config (dict, optional): Environment configuration overrides.
            Supported keys: 'rows' (int), 'cols' (int).
        render_mode (str, optional): Rendering mode ('human', 'rgb_array',
            or None for headless).

    Returns:
        TetrisEnv: A freshly constructed (but not yet reset) environment.
    """
    return TetrisEnv(config=config, render_mode=render_mode)
