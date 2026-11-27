"""Meshes of rectangular cells for potential-field modelling.

:class:`Mesh2D` is a cross-section: ``x`` is distance along the profile and
``z`` is depth below the ground (positive down), both in metres. Every cell
is a prism of rectangular cross section that runs to infinity along strike,
perpendicular to the profile -- the 2-D assumption. Models are flat arrays of
one value per cell, ordered row by row from the top
(``model.reshape(mesh.shape)`` gives ``(nz, nx)``).

:class:`Mesh3D` is a block of rectangular prisms under a map ``x, y``
(east, north), with ``z`` depth as before. Flat models reshape to
``(nz, ny, nx)``: depth slices from the top, each a map with rows of ``y``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Mesh2D", "Mesh3D", "make_mesh", "make_mesh_3d"]


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

    def fraction(self, inside, n_sub: int = 8) -> np.ndarray:
        """Fraction of every cell for which ``inside(x, z)`` is true.

        Estimated on an ``n_sub x n_sub`` grid of points in each cell, for
        shapes with no exact formula (a circle: ``lambda x, z: x**2 +
        (z - 10)**2 < 16``). Use :meth:`rectangle` for rectangles.
        """
        return _subsample(inside, [self.z_edges, self.x_edges], n_sub)

    def core(self, x_min: float, x_max: float, depth: float) -> np.ndarray:
        """Boolean mask of the cells whose centres lie inside a region, flat."""
        x, z = self.centers()
        return (x >= x_min) & (x <= x_max) & (z <= depth)


def _subsample(inside, edges, n_sub: int) -> np.ndarray:
    """Mean of ``inside`` over ``n_sub`` points per cell along each axis.

    ``edges`` is in array order (slowest axis first); ``inside`` receives
    coordinates in the reverse order, ``x`` first.
    """
    t = (np.arange(n_sub) + 0.5) / n_sub
    pts = [(e[:-1, None] + np.diff(e)[:, None] * t).ravel() for e in edges]
    grids = np.meshgrid(*pts, indexing="ij")
    hit = np.asarray(inside(*grids[::-1]), dtype=float)
    shape = []
    for e in edges:
        shape += [e.size - 1, n_sub]
    hit = hit.reshape(shape)
    return hit.mean(axis=tuple(range(1, 2 * len(edges), 2))).ravel()


@dataclass
class Mesh3D:
    """Rectangular prisms between consecutive ``x_edges``, ``y_edges`` and
    ``z_edges``; ``z`` is depth, positive down."""

    x_edges: np.ndarray
    y_edges: np.ndarray
    z_edges: np.ndarray

    def __post_init__(self) -> None:
        for name in ("x_edges", "y_edges", "z_edges"):
            e = np.asarray(getattr(self, name), dtype=float)
            if e.ndim != 1 or e.size < 2 or np.any(np.diff(e) <= 0):
                raise ValueError(f"{name} must be 1-D and strictly increasing")
            setattr(self, name, e)
        if self.z_edges[0] < 0:
            raise ValueError("z is depth: the top of the mesh cannot be above the ground (z < 0)")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.z_edges.size - 1, self.y_edges.size - 1, self.x_edges.size - 1)

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape))

    @property
    def dx(self) -> np.ndarray:
        return np.diff(self.x_edges)

    @property
    def dy(self) -> np.ndarray:
        return np.diff(self.y_edges)

    @property
    def dz(self) -> np.ndarray:
        return np.diff(self.z_edges)

    @property
    def x_centers(self) -> np.ndarray:
        return 0.5 * (self.x_edges[1:] + self.x_edges[:-1])

    @property
    def y_centers(self) -> np.ndarray:
        return 0.5 * (self.y_edges[1:] + self.y_edges[:-1])

    @property
    def z_centers(self) -> np.ndarray:
        return 0.5 * (self.z_edges[1:] + self.z_edges[:-1])

    @property
    def volume(self) -> np.ndarray:
        """Volume of every cell, m^3, flat."""
        return (self.dz[:, None, None] * self.dy[None, :, None] * self.dx[None, None, :]).ravel()

    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``x, y, z`` of every cell centre, flat."""
        Z, Y, X = np.meshgrid(self.z_centers, self.y_centers, self.x_centers, indexing="ij")
        return X.ravel(), Y.ravel(), Z.ravel()

    def box(self, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> np.ndarray:
        """Fraction of every cell inside the box ``[x0, x1] x [y0, y1] x [z0, z1]``, exact."""
        def overlap(e, a, b):
            return np.clip(np.minimum(e[1:], b) - np.maximum(e[:-1], a), 0, None) / np.diff(e)
        fx, fy, fz = overlap(self.x_edges, x0, x1), overlap(self.y_edges, y0, y1), \
            overlap(self.z_edges, z0, z1)
        return (fz[:, None, None] * fy[None, :, None] * fx[None, None, :]).ravel()

    def fraction(self, inside, n_sub: int = 4) -> np.ndarray:
        """Fraction of every cell for which ``inside(x, y, z)`` is true, from
        ``n_sub^3`` points per cell (a sphere: ``lambda x, y, z: x**2 + y**2 +
        (z - 13)**2 < 7.4**2``)."""
        return _subsample(inside, [self.z_edges, self.y_edges, self.x_edges], n_sub)


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


def make_mesh_3d(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    depth: float,
    cell: float = 2.0,
    *,
    cell_z: float | None = None,
    n_pad: int = 4,
    pad_factor: float = 1.4,
) -> Mesh3D:
    """Cells of ``cell`` by ``cell`` by ``cell_z`` (default ``cell``) under the
    rectangle ``[x_min, x_max] x [y_min, y_max]`` down to ``depth``, with
    ``n_pad`` growing padding cells on the four sides and at the bottom.

    A 3-D mesh grows as the cube of its resolution; ``cell`` of a fifth to a
    tenth of the depth of interest is usually as fine as a laptop wants.
    """
    cell_z = cell if cell_z is None else cell_z
    if x_max <= x_min or y_max <= y_min or depth <= 0 or cell <= 0 or cell_z <= 0:
        raise ValueError("need x_max > x_min, y_max > y_min and positive depth and cells")

    def axis(lo, hi, size):
        n = round((hi - lo) / size)
        if not np.isclose(n * size, hi - lo):
            raise ValueError("the core must be a whole number of cells along every axis")
        return lo + size * np.arange(n + 1)

    pad = cell * np.cumsum(pad_factor ** np.arange(1, n_pad + 1))
    pad_z = cell_z * np.cumsum(pad_factor ** np.arange(1, n_pad + 1))
    core_x, core_y, core_z = axis(x_min, x_max, cell), axis(y_min, y_max, cell), \
        axis(0.0, depth, cell_z)
    return Mesh3D(np.concatenate([x_min - pad[::-1], core_x, x_max + pad]),
                  np.concatenate([y_min - pad[::-1], core_y, y_max + pad]),
                  np.concatenate([core_z, depth + pad_z]))
