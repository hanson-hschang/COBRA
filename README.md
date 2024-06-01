<div align=center>
  <h1>Control Oriented BR2 Actuation Model (COBR2)</h1>
</div>

## Project Description

Control Oriented BR2 Actuation Model for a BR2 slender soft arm. The package extend the implementation of Cosserat Rod simulation done by [PyElastica](https://github.com/GazzolaLab/PyElastica).

## How setup development environment

The detail implementation of the following `make` commands are in the `Makefile`.

### Dependency management, installation & packaging

In this project, a Python tool `poetry` is used for dependency management and packaging. To install `poetry` run the following command:

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

In this project, a Python framework `pytest` is used for unit testing. To run the unit tests, run the following command:

```sh
# https://docs.pytest.org/en/8.2.x/index.html
make test
```

### Code formatting

```sh
make formatting
```

### Check type-hinting

A static type checker for Python, `mypy`, is used to checks standard Python programs.

```sh
# https://mypy-lang.org/
make mypy
```

## Related Works

- PyElastica: https://github.com/GazzolaLab/PyElastica
