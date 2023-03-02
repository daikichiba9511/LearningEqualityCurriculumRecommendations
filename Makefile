SHELL=/bin/bash
# -- setting
PYTHON ?= python3.8
RUNCMD ?= poetry run
CUDA ?= cu116

poetry:  ## setup poetry
	curl -sSL https://install.python-poetry.org | python3 -
	poetry config virtualenvs.in-project true
	export PATH="/root/.local/bin:$PATH"
	poetry --version
	touch poetry.toml


setup: poetry ## Install dependencies
	git config --global --add safe.directory /workspace/working/
	poetry install

lint:  ## lint code
	poetry run black --check -l 120 src scripts
	poetry run pflake8 --exit-zero src scripts
	poetry run mypy --show-error-code --pretty src scripts
	poetry run isort -c --diff src scripts

format: ## format code
	poetry run autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive src scripts
	poetry run isort src scripts
	poetry run black -l 120 src scripts

mydotfile:
	git clone --depth 1 git@github.com:daikichiba9511/dotfiles.git ~/dotfiles
	cd ~/dotfiles
	bash setup.sh
	cd -

help:  ## Show all of tasks
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'a
