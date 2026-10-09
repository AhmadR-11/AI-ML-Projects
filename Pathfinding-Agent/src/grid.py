"""Grid data structure and reproducible procedural maze generators.

Foundation module for the Heuristic Graph Pathfinding Agent project.
Provides the Grid representation, cost modeling, movement constraints
(including corner-cutting avoidance for diagonal paths), solvability
validation, and procedural maze generators.
"""

from collections import deque
import math
import random
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# =====================================================================
# PART A — Constants
# =====================================================================

FREE: int = 0
WALL: int = 1
MUD: int = 2

TERRAIN_COST: Dict[int, float] = {
    FREE: 1.0,
    WALL: float("inf"),
    MUD: 5.0,
}

# (row, col) offsets: orthogonal
DIRS_4: List[Tuple[int, int]] = [
    (-1, 0),  # Up
    (1, 0),   # Down
    (0, -1),  # Left
    (0, 1),   # Right
]

# (row, col) offsets: orthogonal followed by diagonal
DIRS_8: List[Tuple[int, int]] = DIRS_4 + [
    (-1, -1),  # Up-Left
    (-1, 1),   # Up-Right
    (1, -1),   # Down-Left
    (1, 1),    # Down-Right
]

SQRT2: float = math.sqrt(2.0)

# Position type alias: (row, col)
Pos = Tuple[int, int]


# =====================================================================
# PART B — Grid Class
# =====================================================================

class Grid:
    """2D grid environment supporting multi-terrain costs and obstacle navigation."""

    def __init__(
        self,
        rows: int,
        cols: int,
        cells: Optional[np.ndarray] = None,
        start: Optional[Pos] = None,
        goal: Optional[Pos] = None,
    ) -> None:
        """Initialize a grid with specified dimensions, terrain, start, and goal.

        Args:
            rows: Number of grid rows (must be positive).
            cols: Number of grid columns (must be positive).
            cells: 2D numpy array of cell terrain types. Defaults to all-FREE np.int8 array.
            start: Start coordinate (row, col). Defaults to (0, 0).
            goal: Goal coordinate (row, col). Defaults to (rows - 1, cols - 1).

        Raises:
            ValueError: If dimensions are non-positive, cells shape mismatches, or
                start/goal are out of bounds or placed on WALL cells.
        """
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Grid dimensions must be positive, got ({rows}, {cols})")

        self.rows: int = rows
        self.cols: int = cols

        if cells is None:
            self.cells: np.ndarray = np.full((rows, cols), FREE, dtype=np.int8)
        else:
            cells_arr = np.asarray(cells, dtype=np.int8)
            if cells_arr.shape != (rows, cols):
                raise ValueError(
                    f"cells array shape {cells_arr.shape} does not match grid dimensions ({rows}, {cols})"
                )
            self.cells = cells_arr

        # Default start and goal locations
        self.start: Pos = start if start is not None else (0, 0)
        self.goal: Pos = goal if goal is not None else (rows - 1, cols - 1)

        # Validate start and goal positions
        self._validate_endpoint(self.start, "start")
        self._validate_endpoint(self.goal, "goal")

    def _validate_endpoint(self, pos: Pos, name: str) -> None:
        """Validate that an endpoint is within bounds and not on a WALL cell."""
        if not self.in_bounds(pos):
            raise ValueError(
                f"{name.capitalize()} position {pos} is out of bounds for grid shape {self.shape}"
            )
        if self.cells[pos[0], pos[1]] == WALL:
            raise ValueError(
                f"{name.capitalize()} position {pos} cannot be placed on a WALL cell"
            )

    @property
    def shape(self) -> Tuple[int, int]:
        """Return dimensions as (rows, cols)."""
        return (self.rows, self.cols)

    def in_bounds(self, pos: Pos) -> bool:
        """Check whether position (row, col) is inside the grid boundary."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_walkable(self, pos: Pos) -> bool:
        """Check if position is in bounds and not an impassable WALL."""
        return self.in_bounds(pos) and self.cells[pos[0], pos[1]] != WALL

    def terrain_cost(self, pos: Pos) -> float:
        """Return the traversal cost of the terrain at pos, or inf if out of bounds."""
        if not self.in_bounds(pos):
            return float("inf")
        cell_type = int(self.cells[pos[0], pos[1]])
        return TERRAIN_COST.get(cell_type, float("inf"))

    def move_cost(self, a: Pos, b: Pos) -> float:
        """Calculate movement cost from pos a to pos b.

        Cost equals terrain_cost(b), multiplied by SQRT2 when moving diagonally.

        Args:
            a: Origin coordinate (row, col).
            b: Destination coordinate (row, col).

        Returns:
            Computed transition cost.
        """
        base_cost = self.terrain_cost(b)
        dr = abs(b[0] - a[0])
        dc = abs(b[1] - a[1])

        # Diagonal movement occurs when both row and column offsets are non-zero (typically 1)
        if dr != 0 and dc != 0:
            return base_cost * SQRT2
        return base_cost

    def neighbors(self, pos: Pos, allow_diagonal: bool = False) -> List[Pos]:
        """Return walkable neighbors in a deterministic fixed order (DIRS order).

        For diagonal moves, enforces NO CORNER CUTTING:
        A diagonal move to (r+dr, c+dc) is valid only if BOTH adjacent orthogonal cells
        (r+dr, c) and (r, c+dc) are walkable.

        Args:
            pos: Current coordinate (row, col).
            allow_diagonal: If True, include valid diagonal moves (DIRS_8).

        Returns:
            List of valid, walkable neighboring coordinates.
        """
        r, c = pos
        result: List[Pos] = []
        dirs = DIRS_8 if allow_diagonal else DIRS_4

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            npos: Pos = (nr, nc)

            if dr == 0 or dc == 0:
                # Orthogonal move
                if self.is_walkable(npos):
                    result.append(npos)
            else:
                # Diagonal move: target must be walkable, and both adjacent orthogonal
                # cells must also be walkable to prevent squeezing/cutting corners.
                ortho_1: Pos = (r + dr, c)
                ortho_2: Pos = (r, c + dc)
                if (
                    self.is_walkable(npos)
                    and self.is_walkable(ortho_1)
                    and self.is_walkable(ortho_2)
                ):
                    result.append(npos)

        return result

    def free_cell_count(self) -> int:
        """Return the number of non-WALL cells in the grid."""
        return int(np.count_nonzero(self.cells != WALL))

    def wall_density(self) -> float:
        """Return ratio of WALL cells to total cells."""
        total = self.rows * self.cols
        return float(np.count_nonzero(self.cells == WALL)) / float(total)

    def copy(self) -> "Grid":
        """Create a deep copy of this Grid instance."""
        return Grid(
            rows=self.rows,
            cols=self.cols,
            cells=self.cells.copy(),
            start=self.start,
            goal=self.goal,
        )

    def set_cell(self, pos: Pos, value: int) -> None:
        """Set terrain value at pos, refusing to place a WALL on start or goal.

        Args:
            pos: Target coordinate (row, col).
            value: Terrain type (FREE, WALL, MUD, etc.).

        Raises:
            ValueError: If pos is out of bounds or placing WALL on start or goal.
        """
        if not self.in_bounds(pos):
            raise ValueError(f"Position {pos} is out of bounds for grid shape {self.shape}")

        if value == WALL and (pos == self.start or pos == self.goal):
            raise ValueError(
                f"Cannot place a WALL on start {self.start} or goal {self.goal}"
            )

        self.cells[pos[0], pos[1]] = value

    def to_ascii(self) -> str:
        """Render grid as human-readable ASCII string.

        Symbols:
            S: start
            G: goal
            #: wall
            ~: mud
            .: free
        """
        lines: List[str] = []
        for r in range(self.rows):
            row_chars: List[str] = []
            for c in range(self.cols):
                pos = (r, c)
                if pos == self.start:
                    row_chars.append("S")
                elif pos == self.goal:
                    row_chars.append("G")
                else:
                    cell_val = self.cells[r, c]
                    if cell_val == WALL:
                        row_chars.append("#")
                    elif cell_val == MUD:
                        row_chars.append("~")
                    elif cell_val == FREE:
                        row_chars.append(".")
                    else:
                        row_chars.append("?")
            lines.append("".join(row_chars))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Grid(rows={self.rows}, cols={self.cols}, start={self.start}, goal={self.goal})"


# =====================================================================
# PART C — Solvability Helper
# =====================================================================

def is_solvable(grid: Grid, allow_diagonal: bool = False) -> bool:
    """Perform a simple BFS reachability check from grid.start to grid.goal.

    NOTE: This is strictly a validation utility for maze generation and integrity
    checks. It is NOT one of the benchmarked pathfinding algorithms evaluated in
    the project.

    Args:
        grid: Grid instance to test.
        allow_diagonal: Whether diagonal moves are considered.

    Returns:
        True if goal is reachable from start, False otherwise.
    """
    if grid.start == grid.goal:
        return True

    queue: deque[Pos] = deque([grid.start])
    visited: set[Pos] = {grid.start}

    while queue:
        current = queue.popleft()
        if current == grid.goal:
            return True

        for neighbor in grid.neighbors(current, allow_diagonal=allow_diagonal):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return False


# =====================================================================
# PART D — Maze Generators
# =====================================================================

def generate_random_grid(
    rows: int,
    cols: int,
    obstacle_density: float,
    seed: int = 0,
    ensure_solvable: bool = True,
    max_attempts: int = 200,
) -> Grid:
    """Generate a grid with uniformly distributed random obstacles.

    Args:
        rows: Grid row count.
        cols: Grid column count.
        obstacle_density: Probability of any cell being a WALL [0.0, 0.9).
        seed: Random seed for reproducibility.
        ensure_solvable: If True, regenerates until start can reach goal.
        max_attempts: Maximum attempts to generate a solvable grid.

    Returns:
        Configured Grid instance.

    Raises:
        ValueError: If density is outside [0.0, 0.9) or dimensions are invalid.
        RuntimeError: If solvable grid cannot be formed within max_attempts.
    """
    if not (0.0 <= obstacle_density < 0.9):
        raise ValueError(
            f"obstacle_density must be in [0.0, 0.9), got {obstacle_density}"
        )
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Grid dimensions must be positive, got ({rows}, {cols})")

    for attempt in range(max_attempts):
        # Derive distinct seed per attempt for reproducible retry sequence
        rng = np.random.default_rng(seed + attempt)
        mask = rng.random((rows, cols)) < obstacle_density
        cells = np.where(mask, WALL, FREE).astype(np.int8)

        # Force start and goal to remain FREE
        cells[0, 0] = FREE
        cells[rows - 1, cols - 1] = FREE

        grid = Grid(rows, cols, cells=cells, start=(0, 0), goal=(rows - 1, cols - 1))

        if not ensure_solvable or is_solvable(grid, allow_diagonal=False):
            return grid

    raise RuntimeError(
        f"Failed to generate a solvable random grid of size ({rows}, {cols}) "
        f"with obstacle density {obstacle_density} after {max_attempts} attempts "
        f"(base seed {seed})."
    )


def generate_perfect_maze(rows: int, cols: int, seed: int = 0) -> Grid:
    """Generate a perfect maze with a single unique solution path using recursive backtracking.

    Uses an iterative DFS with an explicit stack to prevent hitting Python's recursion limit.
    Maze lattice passages require odd dimensions. If rows or cols is even, it is rounded
    UP to the next odd integer so that start (0, 0) and goal (rows - 1, cols - 1) lie on
    passage nodes.

    Args:
        rows: Requested number of rows (rounded up to odd if even).
        cols: Requested number of columns (rounded up to odd if even).
        seed: Seed for the random number generator.

    Returns:
        A Grid instance representing the perfect maze.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Grid dimensions must be positive, got ({rows}, {cols})")

    # Round even dimensions up to next odd number to maintain lattice structure
    actual_rows = rows if rows % 2 == 1 else rows + 1
    actual_cols = cols if cols % 2 == 1 else cols + 1

    # Start with all cells as WALL
    cells = np.full((actual_rows, actual_cols), WALL, dtype=np.int8)

    rng = random.Random(seed)

    # Carve passage on lattice cells at even coordinates (r % 2 == 0, c % 2 == 0)
    start_pos: Pos = (0, 0)
    cells[start_pos[0], start_pos[1]] = FREE
    stack: List[Pos] = [start_pos]

    while stack:
        cr, cc = stack[-1]
        unvisited_neighbors: List[Tuple[Pos, Pos]] = []

        # Check step-2 orthogonal neighbors
        for dr, dc in DIRS_4:
            nr, nc = cr + 2 * dr, cc + 2 * dc
            if 0 <= nr < actual_rows and 0 <= nc < actual_cols:
                if cells[nr, nc] == WALL:
                    wall_between: Pos = (cr + dr, cc + dc)
                    unvisited_neighbors.append(((nr, nc), wall_between))

        if unvisited_neighbors:
            next_cell, wall_cell = rng.choice(unvisited_neighbors)
            # Carve through wall and destination cell
            cells[wall_cell[0], wall_cell[1]] = FREE
            cells[next_cell[0], next_cell[1]] = FREE
            stack.append(next_cell)
        else:
            stack.pop()

    goal_pos: Pos = (actual_rows - 1, actual_cols - 1)
    cells[goal_pos[0], goal_pos[1]] = FREE

    return Grid(
        rows=actual_rows,
        cols=actual_cols,
        cells=cells,
        start=start_pos,
        goal=goal_pos,
    )


def generate_trap_grid(rows: int, cols: int, seed: int = 0) -> Grid:
    """Generate a deterministic grid with a U-shaped / cup-shaped obstacle.

    Builds a cup whose open side faces the start and whose back wall sits between
    the cup and the goal. Always solvable via unobstructed corridors above top
    and below bot.

    Concept sketch:
        ........................
        ......#######...........
        ............#...........
        .S..........#.........G.
        ............#...........
        ......#######...........

    Greedy is lured into the cup by the falling heuristic, wastes expansions
    filling it, then must back out.

    Args:
        rows: Number of grid rows (must be >= 9).
        cols: Number of grid columns (must be >= 15).
        seed: Unused seed kept for interface consistency with other generators.

    Returns:
        A solvable Grid instance containing the U-trap.

    Raises:
        ValueError: If rows < 9 or cols < 15.
    """
    del seed  # Unused; trap obstacle layout is deterministic
    if rows < 9 or cols < 15:
        raise ValueError(
            f"Trap grid requires rows >= 9 and cols >= 15, got ({rows}, {cols})"
        )

    cells = np.full((rows, cols), FREE, dtype=np.int8)

    rm = rows // 2
    half = max(2, rows // 2 - 2)
    top, bot = rm - half, rm + half
    a, b = cols // 2 - 4, cols // 2 + 2

    # Two horizontal arms
    for c in range(a, b + 1):
        cells[top, c] = WALL
        cells[bot, c] = WALL

    # Back wall between cup and goal
    for r in range(top, bot + 1):
        cells[r, b] = WALL

    start: Pos = (rows // 2, 1)
    goal: Pos = (rows // 2, cols - 2)

    grid = Grid(rows=rows, cols=cols, cells=cells, start=start, goal=goal)
    assert is_solvable(grid, allow_diagonal=False), "Generated trap grid must be solvable"
    return grid


def generate_mud_barrier_grid(
    rows: int,
    cols: int,
    band_width: int = 7,
    seed: int = 0,
) -> Grid:
    """Generate a grid with a vertical mud barrier and a cheap top-row corridor.

    Greedy cuts straight through the mud (few steps, high cost) while Dijkstra
    and A* detour via the free top row (more steps, lower cost): the demo of "steps != cost".

    Args:
        rows: Number of grid rows (must be >= 5).
        cols: Number of grid columns (must be >= band_width + 6).
        band_width: Width of the vertical mud band (defaults to 7).
        seed: Unused seed kept for registry signature uniformity.

    Returns:
        A Grid instance with zero walls and a vertical mud barrier.

    Raises:
        ValueError: If dimensions do not satisfy rows >= 5 and cols >= band_width + 6.
    """
    del seed
    if rows < 5 or cols < band_width + 6:
        raise ValueError(
            f"Mud barrier grid requires rows >= 5 and cols >= {band_width + 6}, got ({rows}, {cols})"
        )

    cells = np.full((rows, cols), FREE, dtype=np.int8)
    c0 = cols // 2 - band_width // 2
    c1 = c0 + band_width - 1

    # Vertical band of MUD spanning every row except row 0 (free top corridor)
    for r in range(1, rows):
        for c in range(c0, c1 + 1):
            cells[r, c] = MUD

    start: Pos = (rows // 2, 1)
    goal: Pos = (rows // 2, cols - 2)

    return Grid(rows=rows, cols=cols, cells=cells, start=start, goal=goal)


def generate_unsolvable_grid(rows: int, cols: int, seed: int = 0) -> Grid:
    """Generate a grid guaranteed to be unsolvable by encircling the goal with walls.

    Random obstacles are distributed across the rest of the grid, and all adjacent
    neighbors of the goal (orthogonal and diagonal) are sealed with walls.

    Args:
        rows: Number of rows (must be >= 3).
        cols: Number of columns (must be >= 3).
        seed: Random seed for obstacles outside the enclosure.

    Returns:
        An unsolvable Grid instance.
    """
    if rows < 3 or cols < 3:
        raise ValueError(
            f"Unsolvable grid requires at least 3x3 dimensions, got ({rows}, {cols})"
        )

    rng = np.random.default_rng(seed)
    # Background obstacles with 20% density
    cells = np.where(rng.random((rows, cols)) < 0.2, WALL, FREE).astype(np.int8)

    start_pos: Pos = (0, 0)
    goal_pos: Pos = (rows - 1, cols - 1)

    cells[start_pos[0], start_pos[1]] = FREE
    cells[goal_pos[0], goal_pos[1]] = FREE

    # Seal off all neighbors around the goal
    gr, gc = goal_pos
    for dr, dc in DIRS_8:
        nr, nc = gr + dr, gc + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if (nr, nc) != start_pos and (nr, nc) != goal_pos:
                cells[nr, nc] = WALL

    grid = Grid(rows=rows, cols=cols, cells=cells, start=start_pos, goal=goal_pos)
    assert not is_solvable(grid, allow_diagonal=False), "Expected grid to be unsolvable (4-dir)"
    assert not is_solvable(grid, allow_diagonal=True), "Expected grid to be unsolvable (8-dir)"
    return grid


def add_mud_patches(
    grid: Grid,
    num_patches: int,
    patch_size: Union[int, Tuple[int, int]],
    seed: int = 0,
) -> Grid:
    """Return a COPY of the grid with random rectangular MUD terrain patches.

    Existing WALL cells, start, and goal are never overwritten.

    Args:
        grid: Base Grid to copy and augment.
        num_patches: Number of patches to stamp.
        patch_size: Patch dimensions as an int (square) or (patch_rows, patch_cols).
        seed: Seed for reproducible patch placements.

    Returns:
        New Grid instance with MUD patches added.
    """
    new_grid = grid.copy()
    rng = random.Random(seed)

    if isinstance(patch_size, int):
        p_rows, p_cols = patch_size, patch_size
    else:
        p_rows, p_cols = patch_size

    for _ in range(num_patches):
        # Pick top-left coordinate for the patch
        max_r = max(0, new_grid.rows - p_rows)
        max_c = max(0, new_grid.cols - p_cols)
        r0 = rng.randint(0, max_r)
        c0 = rng.randint(0, max_c)

        for r in range(r0, min(new_grid.rows, r0 + p_rows)):
            for c in range(c0, min(new_grid.cols, c0 + p_cols)):
                pos = (r, c)
                if (
                    pos != new_grid.start
                    and pos != new_grid.goal
                    and new_grid.cells[r, c] != WALL
                ):
                    new_grid.cells[r, c] = MUD

    return new_grid


# Registry mapping readable names to generator callables
MAZE_GENERATORS: Dict[str, Callable[..., Grid]] = {
    "random": generate_random_grid,
    "perfect_maze": generate_perfect_maze,
    "trap": generate_trap_grid,
    "unsolvable": generate_unsolvable_grid,
    "mud_barrier": generate_mud_barrier_grid,
}


# =====================================================================
# PART E — Self-Test Block
# =====================================================================

if __name__ == "__main__":
    def print_grid_info(label: str, g: Grid) -> None:
        """Helper to print grid ASCII visualization and statistics."""
        print(f"\n{'=' * 50}")
        print(f" {label}")
        print(f"{'=' * 50}")
        print(g.to_ascii())
        print(f"Shape:            {g.shape}")
        print(f"Wall Density:     {g.wall_density():.4f}")
        print(f"Free Cell Count:  {g.free_cell_count()}")
        print(f"Solvable (4-dir): {is_solvable(g, allow_diagonal=False)}")
        print(f"Solvable (8-dir): {is_solvable(g, allow_diagonal=True)}")

    # 1. 12x20 random grid with density 0.25 (seed 42)
    rand_grid = generate_random_grid(12, 20, obstacle_density=0.25, seed=42)
    print_grid_info("12x20 Random Grid (density=0.25, seed=42)", rand_grid)

    # 2. 11x21 perfect maze (seed 7)
    perf_maze = generate_perfect_maze(11, 21, seed=7)
    print_grid_info("11x21 Perfect Maze (seed=7)", perf_maze)

    # 3. 12x20 trap grid
    trap_grid = generate_trap_grid(12, 20)
    print_grid_info("12x20 Trap Grid", trap_grid)

    # 4. 15x25 mud barrier grid
    mud_barrier_grid = generate_mud_barrier_grid(15, 25)
    print_grid_info("15x25 Mud Barrier Grid", mud_barrier_grid)

    # 5. 12x20 unsolvable grid
    unsolv_grid = generate_unsolvable_grid(12, 20, seed=42)
    print_grid_info("12x20 Unsolvable Grid", unsolv_grid)

    # 6. 12x20 random grid with mud patches
    mud_grid = add_mud_patches(rand_grid, num_patches=3, patch_size=(3, 4), seed=42)
    print_grid_info("12x20 Random Grid with Mud Patches", mud_grid)

    # -----------------------------------------------------------------
    # Verification Assertions
    # -----------------------------------------------------------------

    # Assertion 1: Same seed produces an identical grid twice
    g_seed_a1 = generate_random_grid(12, 20, 0.25, seed=42)
    g_seed_a2 = generate_random_grid(12, 20, 0.25, seed=42)
    assert np.array_equal(g_seed_a1.cells, g_seed_a2.cells), (
        "Determinism error: Identical seed produced different grids"
    )
    assert g_seed_a1.start == g_seed_a2.start and g_seed_a1.goal == g_seed_a2.goal

    # Assertion 2: Different seeds produce different grids
    g_seed_b = generate_random_grid(12, 20, 0.25, seed=99)
    assert not np.array_equal(g_seed_a1.cells, g_seed_b.cells), (
        "Randomness error: Different seeds produced identical grids"
    )

    # Assertion 3: Unsolvable grid is not solvable
    assert not is_solvable(unsolv_grid, allow_diagonal=False), (
        "Solvability error: Unsolvable grid reported as solvable (4-dir)"
    )
    assert not is_solvable(unsolv_grid, allow_diagonal=True), (
        "Solvability error: Unsolvable grid reported as solvable (8-dir)"
    )

    # Assertion 4: Diagonal move rejected when orthogonal neighbor is wall (Corner-Cutting)
    # Layout:
    # (0, 0) FREE  | (0, 1) FREE
    # (1, 0) WALL  | (1, 1) FREE
    # Moving from (0, 0) to (1, 1): Target (1, 1) is free, but adjacent orthogonal (1, 0) is wall
    test_cells = np.array([
        [FREE, FREE],
        [WALL, FREE],
    ], dtype=np.int8)
    corner_grid = Grid(2, 2, cells=test_cells, start=(0, 0), goal=(1, 1))

    assert (1, 1) not in corner_grid.neighbors((0, 0), allow_diagonal=True), (
        "Corner cutting failure: diagonal move permitted when orthogonal neighbor (1, 0) is a wall"
    )

    # Mirror case: (0, 1) is WALL
    test_cells_mirror = np.array([
        [FREE, WALL],
        [FREE, FREE],
    ], dtype=np.int8)
    corner_grid_mirror = Grid(2, 2, cells=test_cells_mirror, start=(0, 0), goal=(1, 1))
    assert (1, 1) not in corner_grid_mirror.neighbors((0, 0), allow_diagonal=True), (
        "Corner cutting failure: diagonal move permitted when orthogonal neighbor (0, 1) is a wall"
    )

    # Valid diagonal case: both orthogonal neighbors are FREE
    corner_grid.set_cell((1, 0), FREE)
    assert (1, 1) in corner_grid.neighbors((0, 0), allow_diagonal=True), (
        "Movement failure: valid diagonal move rejected when both orthogonal neighbors are free"
    )

    # Additional check: move_cost logic
    assert corner_grid.move_cost((0, 0), (0, 1)) == 1.0
    assert corner_grid.move_cost((0, 0), (1, 1)) == SQRT2
    corner_grid.set_cell((1, 1), MUD)
    assert corner_grid.move_cost((0, 0), (1, 1)) == 5.0 * SQRT2

    # Assertion 5: Trap grid properties
    assert is_solvable(trap_grid, allow_diagonal=False), "Trap grid must be solvable"
    assert trap_grid.start != (0, 0), f"Trap grid start must not be (0, 0), got {trap_grid.start}"

    # Assertion 6: Mud barrier grid properties
    assert mud_barrier_grid.wall_density() == 0.0, "Mud barrier grid must have no walls"
    band_c0 = 25 // 2 - 7 // 2
    band_c1 = band_c0 + 7 - 1
    assert all(mud_barrier_grid.cells[0, c] == FREE for c in range(band_c0, band_c1 + 1)), (
        "Row 0 within the mud band corridor must be FREE"
    )
    assert all(
        mud_barrier_grid.cells[r, c] == MUD
        for r in range(1, 15)
        for c in range(band_c0, band_c1 + 1)
    ), "All rows >= 1 within the mud band must be MUD"

    print("\nAll grid tests passed!")
