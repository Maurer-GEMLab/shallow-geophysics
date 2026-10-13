"""A 2-D cross-section of rectangular cells for potential-field modelling.

``x`` is distance along the profile and ``z`` is depth below the ground
(positive down), both in metres. Every cell is a prism of rectangular cross
section that runs to infinity along strike, perpendicular to the profile --
the 2-D assumption. Models are flat arrays of one value per cell, ordered row
by row from the top (``model.reshape(mesh.shape)`` gives ``(nz, nx)``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Mesh2D", "make_mesh"]


@dataclass
class Mesh2D:
    """Cells between consecutive ``x_edges`` and ``z_edges``."""

    x_edges: np.ndarray
    z_edges: np.ndarray

    def __post_init__(self) -> None:
        self.x_edges = np.asarray(self.x_edges, dtype=float)
        self.z_edges = np.asarray(self.z_edges, dtype=float)
        for name, e in (("x_edges", self.x_edges), ("z_edges", self.z_edges)):
            if e.ndim != 1 or e.size < 2 or np.any(np.diff(e) <= 0):
                raise ValueError(f"{name} must be 1-D and strictly increasing")
        if self.z_edges[0] < 0:
            raise ValueError("z is depth: the top of the mesh cannot be above the ground (z < 0)")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.z_edges.size - 1, self.x_edges.size - 1)

    @property
    def n_cells(self) -> int:
        nz, nx = self.shape
        return nz * nx

    @property
    def dx(self) -> np.ndarray:
        return np.diff(self.x_edges)

    @property
    def dz(self) -> np.ndarray:
        return np.diff(self.z_edges)

    @property
    def x_centers(self) -> np.ndarray:
        return 0.5 * (self.x_edges[1:] + self.x_edges[:-1])

    @property
    def z_centers(self) -> np.ndarray:
        return 0.5 * (self.z_edges[1:] + self.z_edges[:-1])

    @property
    def area(self) -> np.ndarray:
        """Cross-sectional area of every cell, m^2, flat."""
        return np.outer(self.dz, self.dx).ravel()

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """``x, z`` of every cell centre, flat."""
        X, Z = np.meshgrid(self.x_centers, self.z_centers)
        return X.ravel(), Z.ravel()

    def rectangle(self, x0: float, x1: float, z0: float, z1: float) -> np.ndarray:
        """Fraction of every cell inside the rectangle ``[x0, x1] x [z0, z1]``.

        Exact for any rectangle, so a body need not line up with the cells;
        multiply by a property value to put the body in a model.
        """
        fx = np.clip(np.minimum(self.x_edges[1:], x1) - np.maximum(self.x_edges[:-1], x0), 0, None)
        fz = np.clip(np.minimum(self.z_edges[1:], z1) - np.maximum(self.z_edges[:-1], z0), 0, None)
        return np.outer(fz / self.dz, fx / self.dx).ravel()

    def layer(self, z0: float, z1: float) -> np.ndarray:
        """Fraction of every cell between depths ``z0`` and ``z1``."""
        return self.rectangle(self.x_edges[0], self.x_edges[-1], z0, z1)

    def core(self, x_min: float, x_max: float, depth: float) -> np.ndarray:
        """Boolean mask of the cells whose centres lie inside a region, flat."""
        x, z = self.centers()
        return (x >= x_min) & (x <= x_max) & (z <= depth)


def make_mesh(
    x_min: float,
    x_max: float,
    depth: float,
    cell: float = 0.5,
    *,
    n_pad: int = 6,
    pad_factor: float = 1.3,
) -> Mesh2D:
    """Square cells of side ``cell`` under ``[x_min, x_max]`` down to ``depth``.

    ``n_pad`` cells, each ``pad_factor`` times wider than the last, are added
    at both ends and at the bottom. They hold the parts of a smooth model that
    the profile cannot see well, so the core is not forced to explain them;
    keep the profile well inside the core.
    """
    if x_max <= x_min or depth <= 0 or cell <= 0:
        raise ValueError("need x_max > x_min, depth > 0 and cell > 0")
    nx = round((x_max - x_min) / cell)
    nz = round(depth / cell)
    if not np.isclose(nx * cell, x_max - x_min) or not np.isclose(nz * cell, depth):
        raise ValueError("the core width and depth must be whole numbers of cells")
    pad = cell * np.cumsum(pad_factor ** np.arange(1, n_pad + 1))
    core_x = x_min + cell * np.arange(nx + 1)
    x_edges = np.concatenate([x_min - pad[::-1], core_x, x_max + pad])
    z_edges = np.concatenate([cell * np.arange(nz + 1), depth + pad])
    return Mesh2D(x_edges, z_edges)
