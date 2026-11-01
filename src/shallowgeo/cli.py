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

    p_ert = sub.add_parser(
        "ert-clean",
        help="repair a RES2DINV .dat file: integer n, drop readings past the cable",
    )
    p_ert.add_argument("path", type=Path)
    p_ert.add_argument("-e", "--electrodes", type=int, required=True,
                       help="number of electrodes actually in the ground")
    p_ert.add_argument("--first", type=int, default=0,
                       help="index of the first electrode in the ground (default 0)")
    p_ert.add_argument("-o", "--output", type=Path,
                       help="output file (default: <name>_clean.dat next to the input)")

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

    if args.command == "ert-clean":
        from .ert import read_res2dinv

        try:
            raw = read_res2dinv(args.path)
            clean = raw.trim(args.electrodes, first=args.first)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        out = args.output or args.path.with_name(f"{args.path.stem}_clean.dat")
        clean.to_res2dinv(out)
        print(f"{args.path.name}: {len(raw)} readings on {raw.n_electrodes} electrode "
              f"positions; electrodes moved by at most "
              f"{raw.metadata['snapped_max_m'] * 1000:.1f} mm to the "
              f"{raw.spacing:g} m grid")
        print(f"kept {len(clean)} readings on electrodes {args.first + 1}-"
              f"{args.first + args.electrodes}, dropped {len(raw) - len(clean)}")
        cov = clean.coverage()
        n_missing = int(cov["missing"].sum())
        print(f"the kept readings use {len(cov)} array configurations, which fit at "
              f"{int(cov['possible'].sum())} positions on these electrodes; "
              f"{n_missing} of them are not in the file")
        if n_missing:
            print("  (gaps the file had already; trimming did not make them)")
            gaps = cov[cov["missing"] > 0]
            if "dipole" in gaps:
                for dip, g in gaps.groupby("dipole"):
                    counts = ", ".join(f"n={r.sep} {r.measured}/{r.possible}"
                                       for r in g.itertuples())
                    print(f"  dipole {dip} x {clean.spacing:g} m: {counts}")
            else:
                for r in gaps.itertuples():
                    print(f"  offsets {r.offsets}: {r.measured}/{r.possible}")
        print(f"wrote {out}")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
