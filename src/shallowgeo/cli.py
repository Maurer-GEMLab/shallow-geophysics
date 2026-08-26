"""Command-line entry point: ``shallowgeo``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="shallowgeo", description="Near-surface geophysical data tools"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show environment and registered drivers")

    p_id = sub.add_parser("identify", help="report which drivers claim a file")
    p_id.add_argument("path", type=Path)

    p_read = sub.add_parser("read", help="read a file and summarise it")
    p_read.add_argument("path", type=Path)
    p_read.add_argument("--driver", help="force a driver by name")
    p_read.add_argument("--spacing", type=float, help="geophone spacing, metres")
    p_read.add_argument(
        "--source-offset", type=float, help="shot position along line, metres"
    )

    args = parser.parse_args(argv)

    if args.command == "info":
        from . import print_diagnostics

        print_diagnostics()
        return 0

    if args.command == "identify":
        from .drivers import identify

        matches = identify(args.path)
        if not matches:
            print(f"no driver recognises {args.path.name}", file=sys.stderr)
            return 1
        for d in matches:
            print(f"{d.name:<14} {d.description}")
        return 0

    if args.command == "read":
        from .drivers import DriverError, read

        kwargs = {
            k: v
            for k, v in [("spacing", args.spacing), ("source_offset", args.source_offset)]
            if v is not None
        }
        try:
            survey = read(args.path, driver=args.driver, **kwargs)
        except (DriverError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(survey)
        for key, value in survey.metadata.items():
            if not isinstance(value, (dict, list)):
                print(f"  {key}: {value}")
        print("  provenance:")
        for step in survey.provenance:
            print(f"    {step}")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
