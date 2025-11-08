# Repository Guidelines

## Project Structure & Module Organization
- `dataset/`, `dataset_augmented/`, `dataset_split/`: raw, augmented, and split data (train/val/test).
- `models/`: training artifacts (best model, reports, confusion matrix, TFLite export).
- Scripts: `record_audio.py`, `augment_audio.py`, `prepare_dataset.py`, `train_model.py`.
- Keep experimental notebooks or scratch code in `experiments/` (if added) and avoid committing large audio.

## Build, Test, and Development Commands
- Environment (Windows example): `python -m venv .venv && .\.venv\Scripts\Activate`.
- Install deps: `pip install -r requirements.txt`.
  - If missing, install: `pip install pyaudio pygame librosa soundfile numpy tensorflow tqdm matplotlib seaborn scikit-learn`.
- Prepare dirs: `python prepare_dataset.py create`.
- Record samples: `python record_audio.py`.
- Augment data: `python augment_audio.py`.
- Split sets: `python prepare_dataset.py split`.
- Train/export: `python train_model.py` (writes to `models/`).

## Coding Style & Naming Conventions
- Follow PEP 8; 4‑space indentation; max line length 88.
- `snake_case` for files/functions, `PascalCase` for classes, UPPER_CASE for constants.
- Add type hints and docstrings to new/changed functions.
- Keep scripts idempotent and path‑safe; prefer `pathlib`.

## Testing Guidelines
- Prefer `pytest` for unit tests; name files `test_*.py` near code or under `tests/`.
- Smoke tests: validate CLI flows with small dummy WAVs; assert outputs/paths exist (e.g., under `dataset_split/` and `models/`).
- Heavy training is integration-level: use a tiny subset to keep tests fast.
- Optional coverage: `pytest --maxfail=1 -q` (add `pytest-cov` if needed).

## Commit & Pull Request Guidelines
- Commits: imperative mood, focused changes (e.g., "Add MFCC augmentation option").
- PRs: description, rationale, steps to reproduce, sample logs/metrics; link issues.
- Do not commit large raw audio; keep only minimal samples. Add or respect `.gitignore` for dataset outputs.
- Before merging: ensure scripts run end‑to‑end and outputs land in the documented folders.

## Data & Security Notes
- Record in a quiet environment; verify microphone permissions.
- Anonymize or avoid sensitive recordings; remove accidental captures before pushing.
