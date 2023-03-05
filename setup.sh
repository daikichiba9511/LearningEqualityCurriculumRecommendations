#!/bin/bash
poetry run python -m pip install --upgrade pip
poetry install
poetry run python -m pip install cudf-cu11 dask-cudf-cu11 --extra-index-url=https://pypi.nvidia.com
poetry run python -m pip install cuml-cu11 --extra-index-url=https://pypi.nvidia.com
