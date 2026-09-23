# Examples

Worked, runnable examples for the course. Two kinds live here:

- `data/` — example **datasets**: complete, documented field surveys that the
  notebooks and scripts read. See [data/README.md](data/README.md) for the
  layout every dataset must follow before it is added.
- `scripts/` — short, plain-Python end-to-end runs (read, pick, fit, plot)
  for people who prefer a terminal to a notebook. Each script takes a
  dataset directory as its first argument.

Interactive teaching material is in [`../notebooks/`](../notebooks/), which
is Colab-friendly and imports this package.

## Difference from `tests/data/`

`tests/data/` holds *format samples*: single trimmed files whose job is to
prove a driver still parses a vendor layout. They are small on purpose and
carry no scientific story. A dataset here is the opposite: many files from
one survey, with the geometry, site and acquisition notes needed to
interpret them.

## Adding a dataset

1. Create `data/<slug>/` following the layout in `data/README.md`.
2. Write the `dataset.md` sidecar first. If you cannot fill in the source
   type, spread geometry, and record length, the data are not ready.
3. Add a script or notebook that reads it end to end.
4. Keep the total under a few megabytes. Anything larger goes on a release
   asset or a data repository, with a `fetch.py` in the dataset folder that
   downloads it and checks a hash.
