# Swim Analysis

Local tool to analyze swim footage and extract stroke rate, stroke count,
distance per stroke, underwater time/distance, and turn time.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project structure

- `src/` — core pipeline (pose tracking, trajectory smoothing, metrics, calibration, annotation)
- `data/raw/` — original uploaded footage (not tracked in git)
- `data/processed/` — annotated output videos (not tracked in git)
- `data/samples/` — small sample clips for demos/testing (tracked in git)
- `notebooks/` — exploration and validation notebooks
- `config.yaml` — pipeline parameters (smoothing, thresholds, calibration)

## Status

Work in progress.
