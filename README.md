<div align=center>
  <h1>COBRA - Control Oriented Bending and Rotational Actuation Model</h1>
</div>

## Project Description

Control Oriented Actuation Model for a Bending and/or Rotational slender soft arms.
The package extend the implementation of Cosserat Rod simulation done by [PyElastica](https://github.com/GazzolaLab/PyElastica).

## How setup development environment

The detail implementation of the following `make` commands are in the `Makefile`.
Here is a [Makefile tutorial](https://makefiletutorial.com/).

### Dependency management, installation & packaging

In this project, a Python tool [`poetry`](https://python-poetry.org/) is used for dependency management and packaging.
To install, run the following command:

```sh
# https://python-poetry.org/docs/#installing-with-the-official-installer
make poetry-download
```

To remove the poetry, simply run `make poetry-remove`.

To install the dependencies for development, run the following command:

```sh
# https://python-poetry.org/docs/cli/#install
make install
```
This generates the `poetry.lock` and `requirements.txt` and installs packages accordingly.

To install Git hook scripts tool to identify simple issues before submission to code review, run the following command:

```sh
# https://pre-commit.com/#install
make pre-commit-install
```

###  Unittests

In this project, a Python framework [`pytest`](https://docs.pytest.org/en/8.2.x/index.html) is used for unit testing.
To run unit tests, run the following command:

```sh
make pytest
```

### Code formatting

This project uses [`isort`](https://pycqa.github.io/isort/) to sort and organize imported packages, and [`black`](https://black.readthedocs.io/en/stable/) to enforce a consistent code style across the Python codebase.
To format the codebase, run the following command:

```sh
make formatting
```

### Check type-hinting

A static type checker for Python, [`mypy`](https://mypy-lang.org/), is used to check standard Python programs.
To check type hinting, run the following command:

```sh
make mypy
```

### Install the cobra as an editable package

To use the `cobra` as an package, run the following command:

```sh
pip install -e .
```

## Related Works

- PyElastica: https://github.com/GazzolaLab/PyElastica
