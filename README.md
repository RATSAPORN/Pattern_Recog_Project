# Image Captioning with VMamba

A deep learning project for image captioning using the VMamba architecture.

---

## Project Structure

```
image_captioning_with_vmamba/
├── data/                  # Dataset files
├── notebooks/             # Exploratory notebooks
│   └── Data_exploration_for_Image_captioning_model.ipynb
├── src/
│   ├── data/              # Data download and preprocessing scripts
│   │   ├── make_data.py
│   │   └── build_features.py
│   ├── models/            # Model architecture, training, and inference
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── train.py
│   │   └── predict.py
│   └── __init__.py
├── pyproject.toml         # Project metadata and dependencies (Poetry)
├── poetry.lock            # Exact dependency versions — always commit this
└── README.md
```

---

## Setup & Installation

This project uses **Poetry** for dependency management.

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)

Install Poetry if you don't have it:

```bash
pip install poetry
```

### Install dependencies

```bash
poetry install
```

Poetry will automatically create a virtual environment and install all dependencies at the exact versions pinned in `poetry.lock`.

---

## Updating Dependencies

### 1. Add a new package

```bash
poetry add <package-name>

# For dev-only dependencies (e.g. testing, linting)
poetry add --group dev <package-name>
```

### 2. Remove a package

```bash
poetry remove <package-name>
```

### 3. Commit the changes

Always commit `pyproject.toml` and `poetry.lock` together:

```bash
git add pyproject.toml poetry.lock
git commit -m "chore: update dependencies"
```

> **Important:** Never edit `pyproject.toml` or `poetry.lock` by hand. Always use `poetry add` / `poetry remove` so the lock file stays consistent.
