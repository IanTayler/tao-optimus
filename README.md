# Optimus

Optimus is a python library meant for educational purposes. The idea is to build optimization algorithms from the ground up using only `numpy` and without adding some of the more complicated performance enhancements that would make the code harder to understand.

This library was built while working on an optimization course dictated at Uruguay's Universidad de la República called Theory and Algorithms for Optimization ([link](https://eva.fing.edu.uy/course/view.php?id=963&section=0#tabs-tree-start)). Often shortened to TAO. **Note**: the course is in Spanish, but the entirety of the code and documentation in this repo is in English.

## Installation

This repository uses `poetry` for dependency handling. To install all dependencies run `poetry install`. Make sure you have python 3.9 installed.

For a guide on how to install poetry itself, check out the official [poetry documentation](https://python-poetry.org/docs/).

## Optimus

Manually implements iterative optimization algorithms as a learning exercise. Go to the internal [README](/optimus/README.md) for more information.

Also, you can run `poetry run pdoc optimus` to get reference documentation of the `optimus` library, autogenerated from the library's docstrings.
