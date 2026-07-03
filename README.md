# biohub-cell-tracking-during-development

Exploration and tooling for the Kaggle competition
[**biohub-cell-tracking-during-development**](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development):
tracking cells through time in 3D light-sheet microscopy volumes.

## Data

Each sample is a pair under `data/train/<id>`:

- `<id>.zarr` — image volume, shape `(T, Z, Y, X)`, `uint16` (Zarr v3).
- `<id>.geff` — ground-truth tracking graph. Nodes are cell centroids `(t, z, y, x)`;
  edges link a cell's `source_id` at `t` to its `target_id` at `t+1`.

`data/` is gitignored. Download samples from the competition with the Kaggle CLI
(requires an API token at `~/.kaggle/kaggle.json` and accepting the competition rules).

## Layout

- `inference/viewer.ipynb` — interactive viewer: renders a volume at `t` beside `t+1`,
  overlays the annotations, and colors each cell track consistently. A `follow` dropdown
  locks onto one track and steps its z-slice through time. See the notebook's intro cell.
- `nbs/` — scratch notebooks (gitignored).

## Setup

```bash
pip install "zarr>=3" geff networkx pandas numpy matplotlib ipywidgets
```