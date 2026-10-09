"""Heuristic distance functions for grid pathfinding.

Provides admissible and consistent heuristics for 4-direction and 8-direction
movement under the project's cost model (minimum terrain cost 1.0, diagonal
move cost SQRT2 * terrain_cost). All heuristic functions are pure, symmetric,
and return floating-point estimates between any two grid coordinates.
"""

import math
from typing import Callable, Dict

from src.grid import Pos, SQRT2


def manhattan(a: Pos, b: Pos) -> float:
    """Compute Manhattan (L1) distance: |dr| + |dc|.

    Admissibility:
        Admissible ONLY for 4-direction (orthogonal) movement on unit-or-higher
        cost terrain. It can overestimate true path cost when diagonal movement
        is enabled (since a diagonal step covers |dr|=1, |dc|=1 with cost SQRT2 < 2.0).
        Consequently, using Manhattan with 8-direction movement can make A* non-optimal.

    Tightness:
        Tightest admissible heuristic on a 4-direction uniform grid. Part of the
        tightness chain: zero <= chebyshev <= euclidean <= octile <= manhattan.

    Args:
        a: Source position (row, col).
        b: Destination position (row, col).

    Returns:
        Estimated distance as a float.
    """
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return float(dr + dc)


def euclidean(a: Pos, b: Pos) -> float:
    """Compute Euclidean (L2) straight-line distance: sqrt(dr² + dc²).

    Admissibility:
        Admissible for BOTH 4-direction and 8-direction movement under the
        project's cost model. Since straight-line Euclidean distance is the shortest
        possible metric distance in continuous space and minimum step cost is 1.0
        (or SQRT2 diagonally), it never overestimates the true travel cost.

    Tightness:
        Strictly lower than or equal to Octile distance on 8-direction grids.
        Part of the tightness chain: zero <= chebyshev <= euclidean <= octile <= manhattan.

    Args:
        a: Source position (row, col).
        b: Destination position (row, col).

    Returns:
        Estimated distance as a float.
    """
    dr = a[0] - b[0]
    dc = a[1] - b[1]
    return float(math.hypot(dr, dc))


def octile(a: Pos, b: Pos) -> float:
    """Compute Octile distance: max(dr, dc) + (SQRT2 - 1) * min(dr, dc).

    Admissibility:
        Admissible for BOTH 4-direction and 8-direction movement. It is exact
        (perfectly informed) on an empty 8-direction grid where diagonal moves cost
        SQRT2 and orthogonal moves cost 1.0. It never overestimates true travel cost.

    Recommendation:
        Recommended default heuristic for 8-direction movement because it is the
        tightest possible admissible distance metric on an 8-direction grid.
        Part of the tightness chain: zero <= chebyshev <= euclidean <= octile <= manhattan.

    Args:
        a: Source position (row, col).
        b: Destination position (row, col).

    Returns:
        Estimated distance as a float.
    """
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return float(max(dr, dc) + (SQRT2 - 1.0) * min(dr, dc))


def chebyshev(a: Pos, b: Pos) -> float:
    """Compute Chebyshev (L-infinity) distance: max(dr, dc).

    Admissibility:
        Admissible for BOTH 4-direction and 8-direction movement. While it assumes
        diagonal moves cost 1.0 (underestimating the project's SQRT2 diagonal cost),
        it remains strictly admissible and consistent.

    Tightness:
        Looser than Euclidean and Octile distances. Part of the tightness chain:
        zero <= chebyshev <= euclidean <= octile <= manhattan.

    Args:
        a: Source position (row, col).
        b: Destination position (row, col).

    Returns:
        Estimated distance as a float.
    """
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return float(max(dr, dc))


def zero(a: Pos, b: Pos) -> float:
    """Compute null heuristic: always returns 0.0.

    Admissibility:
        Trivially admissible for all grid topologies and cost models. Reduces
        A* search to standard Dijkstra's algorithm. Frequently used for baseline
        comparisons and algorithm correctness validation.

    Tightness:
        Bottom baseline of the tightness chain:
        zero <= chebyshev <= euclidean <= octile <= manhattan.

    Args:
        a: Source position (row, col).
        b: Destination position (row, col).

    Returns:
        0.0 as a float.
    """
    del a, b
    return 0.0


# Registry of available heuristics
HEURISTICS: Dict[str, Callable[[Pos, Pos], float]] = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "octile": octile,
    "chebyshev": chebyshev,
    "zero": zero,
}


def get_heuristic(name: str) -> Callable[[Pos, Pos], float]:
    """Retrieve a heuristic function by its registered name.

    Args:
        name: Name of the heuristic (case-sensitive).

    Returns:
        The corresponding heuristic callable.

    Raises:
        KeyError: If name is not recognized, listing valid alternatives.
    """
    if name not in HEURISTICS:
        valid_keys = ", ".join(repr(k) for k in HEURISTICS.keys())
        raise KeyError(
            f"Unknown heuristic {name!r}. Available heuristics are: {valid_keys}"
        )
    return HEURISTICS[name]


def default_heuristic_name(allow_diagonal: bool) -> str:
    """Return the recommended default heuristic name based on movement model.

    Args:
        allow_diagonal: True if 8-direction movement is enabled, False for 4-direction.

    Returns:
        "octile" when allow_diagonal is True, else "manhattan".
    """
    return "octile" if allow_diagonal else "manhattan"


# =====================================================================
# Self-Test Block
# =====================================================================

if __name__ == "__main__":
    coords = [(r, c) for r in range(6) for c in range(6)]

    # 1. Identity h(a, a) == 0 and symmetry h(a, b) == h(b, a)
    for name, h_func in HEURISTICS.items():
        for pos in coords:
            assert h_func(pos, pos) == 0.0, f"{name}: h(a, a) must be 0 for {pos}"

        for i, a in enumerate(coords):
            for b in coords[i:]:
                val_ab = h_func(a, b)
                val_ba = h_func(b, a)
                assert math.isclose(val_ab, val_ba, rel_tol=1e-9, abs_tol=1e-9), (
                    f"{name} symmetry violated between {a} and {b}: {val_ab} != {val_ba}"
                )

    # 2. Known values for a=(0, 0), b=(3, 4)
    p0 = (0, 0)
    p1 = (3, 4)
    assert manhattan(p0, p1) == 7.0, f"Manhattan expected 7.0, got {manhattan(p0, p1)}"
    assert euclidean(p0, p1) == 5.0, f"Euclidean expected 5.0, got {euclidean(p0, p1)}"
    assert chebyshev(p0, p1) == 4.0, f"Chebyshev expected 4.0, got {chebyshev(p0, p1)}"
    expected_octile = 4.0 + (SQRT2 - 1.0) * 3.0
    assert math.isclose(octile(p0, p1), expected_octile, rel_tol=1e-9, abs_tol=1e-9), (
        f"Octile expected {expected_octile}, got {octile(p0, p1)}"
    )

    # 3. Tightness chain: zero <= chebyshev <= euclidean <= octile <= manhattan
    tol = 1e-9
    for a in coords:
        for b in coords:
            h_zero = zero(a, b)
            h_cheb = chebyshev(a, b)
            h_eucl = euclidean(a, b)
            h_octi = octile(a, b)
            h_manh = manhattan(a, b)

            assert h_zero <= h_cheb + tol, (
                f"Tightness violation: zero ({h_zero}) > chebyshev ({h_cheb}) for {a}->{b}"
            )
            assert h_cheb <= h_eucl + tol, (
                f"Tightness violation: chebyshev ({h_cheb}) > euclidean ({h_eucl}) for {a}->{b}"
            )
            assert h_eucl <= h_octi + tol, (
                f"Tightness violation: euclidean ({h_eucl}) > octile ({h_octi}) for {a}->{b}"
            )
            assert h_octi <= h_manh + tol, (
                f"Tightness violation: octile ({h_octi}) > manhattan ({h_manh}) for {a}->{b}"
            )

    # 4. Unknown name raises KeyError
    try:
        get_heuristic("invalid_heuristic_name")
        assert False, "Expected KeyError for invalid heuristic name"
    except KeyError as exc:
        assert "Available heuristics are" in str(exc), "Error message missing valid list"

    # 5. Default heuristic selection
    assert default_heuristic_name(False) == "manhattan"
    assert default_heuristic_name(True) == "octile"

    print("All heuristic tests passed!")
