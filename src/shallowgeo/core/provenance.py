"""Append-only processing lineage.

Every ``Survey`` carries one of these. Readers record the ingest step; every
correction and transform appends. The point is that "raw" versus "corrected"
is never ambiguous -- a frequent and expensive confusion with field data,
especially for gravity, where a number alone does not tell you which of the
half-dozen standard corrections have already been applied.

Modeled on ObsPy's ``Trace.stats.processing``, but structured rather than
free-text so it can be serialized and queried.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Step:
    """One recorded operation."""

    operation: str
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        )
    )
    software: str = "shallowgeo"

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(self.parameters.items()))
        return f"{self.timestamp} {self.operation}({args})"


class Provenance:
    """Ordered, append-only sequence of :class:`Step`."""

    __slots__ = ("_steps",)

    def __init__(self, steps: list[Step] | None = None) -> None:
        self._steps: list[Step] = list(steps or [])

    def record(self, operation: str, **parameters: Any) -> Step:
        step = Step(operation=operation, parameters=parameters)
        self._steps.append(step)
        return step

    def applied(self, operation: str) -> bool:
        """Whether *operation* has already been recorded.

        Corrections use this to refuse to run twice -- applying a drift or
        diurnal correction a second time is silent and destructive.
        """
        return any(s.operation == operation for s in self._steps)

    def copy(self) -> Provenance:
        return Provenance(list(self._steps))

    def to_list(self) -> list[dict[str, Any]]:
        return [
            {
                "operation": s.operation,
                "parameters": s.parameters,
                "timestamp": s.timestamp,
                "software": s.software,
            }
            for s in self._steps
        ]

    @classmethod
    def from_list(cls, records: list[dict[str, Any]]) -> Provenance:
        return cls([Step(**r) for r in records])

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self):
        return iter(self._steps)

    def __getitem__(self, i):
        return self._steps[i]

    def __repr__(self) -> str:
        return f"<Provenance: {len(self._steps)} step(s)>"
