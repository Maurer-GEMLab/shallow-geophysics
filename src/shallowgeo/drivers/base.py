"""GDAL-style driver registry.

A driver is a small object that knows how to recognise one vendor format and
turn it into a Layer 0 :class:`~shallowgeo.core.survey.Survey`. Drivers are
discovered through the ``shallowgeo.drivers`` entry-point group, so an
external package can add support for an instrument without patching this one::

    [project.entry-points."shallowgeo.drivers"]
    my-instrument = "mypkg.reader:driver"

Recognition is content-sniffing first, extension second. Vendor software is
careless about extensions -- Geode files turn up as ``.dat``, ``.sg2``, and
``.seg2`` interchangeably -- so an extension match alone is treated as a weak
signal, used only to order candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..core.survey import Survey


class DriverError(RuntimeError):
    """Raised when no driver matches, or a matching driver fails to read."""


@dataclass
class Driver:
    """One format reader.

    Attributes
    ----------
    name
        Stable identifier, used for ``read(..., driver="cg5")``.
    can_open
        ``(Path) -> bool``. Must be cheap and must not raise on unrelated
        files; sniff a header, do not parse the body.
    read
        ``(Path, **kwargs) -> Survey``.
    extensions
        Lowercase, dot-prefixed. Advisory only.
    """

    name: str
    description: str
    can_open: Callable[[Path], bool]
    read: Callable[..., Survey]
    extensions: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    vendor: str = ""
    instrument: str = ""
    notes: str = ""
    _extras: dict[str, Any] = field(default_factory=dict, repr=False)

    def matches_extension(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions


class _Registry:
    """Process-wide driver table, populated lazily on first use."""

    def __init__(self) -> None:
        self._drivers: dict[str, Driver] = {}
        self._loaded = False

    # -- population ---------------------------------------------------------

    def register(self, driver: Driver, *, replace: bool = False) -> Driver:
        if driver.name in self._drivers and not replace:
            raise ValueError(
                f"driver {driver.name!r} already registered; "
                "pass replace=True to override"
            )
        self._drivers[driver.name] = driver
        return driver

    def _load_entry_points(self) -> None:
        if self._loaded:
            return
        # Set the flag first: a driver whose import fails should not cause the
        # whole registry to be re-scanned on every subsequent call.
        self._loaded = True
        from importlib.metadata import entry_points

        for ep in entry_points(group="shallowgeo.drivers"):
            try:
                driver = ep.load()
            except Exception as exc:  # pragma: no cover - defensive
                import warnings

                warnings.warn(
                    f"failed to load driver entry point {ep.name!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            self._drivers.setdefault(driver.name, driver)

    # -- lookup -------------------------------------------------------------

    def all(self) -> list[Driver]:
        self._load_entry_points()
        return sorted(self._drivers.values(), key=lambda d: d.name)

    def get(self, name: str) -> Driver:
        self._load_entry_points()
        try:
            return self._drivers[name]
        except KeyError:
            known = ", ".join(sorted(self._drivers)) or "(none)"
            raise DriverError(
                f"no driver named {name!r}. Registered: {known}"
            ) from None

    def identify(self, path: str | Path) -> list[Driver]:
        """Every driver that claims *path*, extension matches ordered first."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = self.all()
        candidates.sort(key=lambda d: not d.matches_extension(path))
        out = []
        for d in candidates:
            try:
                if d.can_open(path):
                    out.append(d)
            except Exception:
                # A driver that blows up while sniffing an unrelated file is a
                # bug in that driver, not a reason to fail the whole lookup.
                continue
        return out


registry = _Registry()


def register(driver: Driver, *, replace: bool = False) -> Driver:
    return registry.register(driver, replace=replace)


def identify(path: str | Path) -> list[Driver]:
    """Drivers that recognise *path*, best guess first."""
    return registry.identify(path)


def read(path: str | Path, driver: str | None = None, **kwargs: Any) -> Survey:
    """Read *path* into a :class:`~shallowgeo.core.survey.Survey`.

    Parameters
    ----------
    driver
        Force a specific driver by name. Otherwise the registry sniffs, and
        raises if nothing matches.

    Remaining keyword arguments are passed to the driver, which is how
    geometry that the file does not carry gets supplied -- for example
    ``read("LINE1.DAT", spacing=2.0, source_offset=-1.0)``.
    """
    path = Path(path)
    if driver is not None:
        return registry.get(driver).read(path, **kwargs)

    matches = identify(path)
    if not matches:
        raise DriverError(
            f"no registered driver recognises {path.name!r}. "
            f"Available: {', '.join(d.name for d in registry.all()) or '(none)'}. "
            "Force one with read(..., driver='name')."
        )
    return matches[0].read(path, **kwargs)


def _sniff(path: Path, n: int = 512) -> bytes:
    """First *n* bytes, for use in ``can_open``."""
    try:
        with open(path, "rb") as fh:
            return fh.read(n)
    except OSError:
        return b""


def _sniff_text(path: Path, n_lines: int = 40) -> list[str]:
    """First *n_lines* decoded leniently -- vendor ASCII is rarely clean UTF-8."""
    try:
        with open(path, "r", encoding="latin-1", errors="replace") as fh:
            return [next(fh, "") for _ in range(n_lines)]
    except OSError:
        return []


def _iter_lines(paths: Iterable[str]) -> Iterable[str]:  # pragma: no cover
    for p in paths:
        yield from _sniff_text(Path(p))
