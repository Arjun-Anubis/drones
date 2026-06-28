# drones

This project contains notebooks and scripts for exploring drone maintenance and failure prediction data.

## Requirements

This project uses UV for dependency and environment management.

### Install UV

If UV is not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create and activate the environment

From the project root:

```bash
cd /home/anubi/drones
uv sync
```

This will create the virtual environment and install the dependencies listed in pyproject.toml.

## Run the project

### Run Python scripts

```bash
uv run python main.py
```

### Run notebooks

Start Jupyter Lab with:

```bash
uv run jupyter lab
```

If you want to use the environment in VS Code notebooks, install the kernel once:

```bash
uv run python -m ipykernel install --user --name drones
```

## Project structure

- data/ - input datasets
- notebooks and .ipynb files - exploratory analysis and baselines
- pyproject.toml - UV dependency configuration
- uv.lock - locked dependency versions

## Notes

- Use `uv run` for commands so they run inside the project environment.
- If you add new dependencies, update them with:

```bash
uv add <package-name>
```
