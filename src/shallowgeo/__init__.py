"""shallow-geophysics -- unified near-surface geophysical data and modeling.

Read any supported instrument's export into one common survey object::

    import shallowgeo as sg

    refraction = sg.read("LINE1.DAT", spacing=2.0, source_offset=-1.0)
    gravity    = sg.read("GRAV.TXT")
    print(sg.drivers.registry.all())

Scope of this release: seismic refraction, MASW and passive surface wave,
ground gravity, and ground magnetics.
"""

from . import core, drivers
from .core import (
    Geometry,
    PointSurvey,
    Provenance,
    SeismicSurvey,
    SpatialRef,
    Survey,
    local_grid,
)
from .drivers import identify, read

__version__ = "0.0.1.dev0"

__all__ = [
    "core", "drivers", "read", "identify",
    "Geometry", "SpatialRef", "local_grid", "Provenance",
    "Survey", "SeismicSurvey", "PointSurvey",
    "print_diagnostics", "__version__",
]


def print_diagnostics() -> None:
    """Report the environment and registered drivers.

    The first thing to run on a student's laptop, and the first thing to paste
    into a bug report.
    """
    import platform
    import sys

    print(f"shallow-geophysics {__version__}")
    print(f"Python {sys.version.split()[0]} on {platform.platform()}")

    print("\nOptional stack:")
    for name, extra in [
        ("numpy", None), ("pandas", None), ("pyproj", None),
        ("obspy", "seismic"), ("discretize", "model"), ("xarray", "model"),
        ("simpeg", "invert"), ("pygimli", "invert"),
        ("disba", "masw"), ("evodcinv", "masw"),
    ]:
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "?")
            print(f"  [ok]      {name:<12} {version}")
        except ImportError:
            hint = f"  (extra: {extra})" if extra else ""
            print(f"  [missing] {name:<12}{hint}")

    print("\nRegistered drivers:")
    for d in drivers.registry.all():
        print(f"  {d.name:<14} {d.description}")
        if d.notes:
            print(f"                 note: {d.notes}")
