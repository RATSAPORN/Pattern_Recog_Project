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
├── requirements.txt       # Exported dependencies (for pip/Docker compatibility)
├── poetry.lock            # lock file to view the dependencies log
└── README.md
```

---

## Setup & Installation

This project uses **Poetry** for dependency management. A `requirements.txt` is also maintained for compatibility with pip-based environments (e.g. Docker, CI).

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

This creates a virtual environment automatically and installs all dependencies pinned in `poetry.lock`.

> If you prefer using a plain virtual environment with pip instead:
>
> ```bash
> python -m venv venv
> source venv/bin/activate      # On Windows: venv\Scripts\activate
> pip install -r requirements.txt
> ```

---

## Updating Dependencies

We use Poetry as the source of truth for dependencies. After adding or changing any package, you must regenerate `requirements.txt` so both stay in sync.

### 1. Add a new package

```bash
poetry add <package-name>

# For dev-only dependencies (e.g. testing, linting)
poetry add --group dev <package-name>
```

### 2. Sync `requirements.txt`

After any dependency change, regenerate `requirements.txt`:

```bash
poetry export -f requirements.txt --without-hashes -o requirements.txt
```

### 3. Commit both files

Always commit `pyproject.toml`, `poetry.lock`, and `requirements.txt` together:

```bash
git add pyproject.toml poetry.lock requirements.txt
git commit -m "chore: update dependencies"
```

> **Important:** Never edit `requirements.txt` by hand — it is auto-generated from Poetry. All dependency changes should go through `pyproject.toml` via `poetry add` or `poetry remove`.
