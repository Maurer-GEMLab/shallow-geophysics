# Format sample corpus

Regression fixtures: small, real files from real instruments, used to prove the
drivers keep working as vendor firmware changes.

Currently holds the two G-857 layouts (`g857_exported.asc`,
`g857_manual.txt`), written from layouts specified by the instrument owner
rather than exported from hardware — so header wording on a given firmware is
still unconfirmed. Everything else is validated only against the synthetic
SEG-2 fixtures in `tests/conftest.py`, which encode our *assumptions* about
each format rather than its reality.

## Contributing a sample

Good samples are small (trim to a few traces or readings), from a stated
instrument and firmware version, and carry no survey data you would mind
publishing — this repository is public.

1. Add the file here, with a sibling `<name>.md` recording instrument model,
   firmware version, acquisition software, and anything unusual.
2. Add a test in `tests/test_drivers.py` asserting the values you know to be
   correct — not just that parsing succeeds.
3. Commit normally — this directory is explicitly un-ignored. `.gitignore`
   excludes `/data/` at the repo root for bulk field data, but not here. If a
   sample seems to vanish on commit, check `git check-ignore -v <path>`.

## What is most useful right now

| Priority | File | Why |
|---|---|---|
| **High** | Geode SEG-2 with geometry entered | Confirms header convention |
| **High** | Geode SEG-2 with geometry *not* entered | The common teaching-lab case |
| Medium | G-857 dump from the real instrument | Confirms firmware header wording |
| Medium | ATOM-1C SEG-2 export | Confirms the GPS header key |
| Medium | CG-5 `.TXT` dump | Confirms column labels per firmware |
| Low | ATOM-1C native `.ATM` | Would need reverse-engineering |
