.PHONY: help setup download-data format lint train-cls train-reg train-mtl train-all clean

PYTHON := uv run python
DATA_URL := https://github.com/marcin119a/data/raw/refs/heads/main/data_gsn.zip

help:
	@echo "Available commands:"
	@echo "  make setup          - Install project dependencies via uv"
	@echo "  make download-data  - Download and extract the GSN dataset"
	@echo "  make format         - Auto-format code and fix linting errors"
	@echo "  make lint           - Check code for linting and formatting issues"
	@echo "  make train-cls      - Train Exp 1: Classification Only"
	@echo "  make train-reg      - Train Exp 2: Regression Only"
	@echo "  make train-mtl      - Train Exp 3: Multitask Target (Best)"
	@echo "  make train-all      - Run all three training experiments sequentially"
	@echo "  make clean          - Remove cache and temporary files"

setup:
	uv sync --all-extras

download-data:
	mkdir -p data/
	curl -L -A "Mozilla/5.0" -o data/data_gsn.zip $(DATA_URL)
	unzip -o -q data/data_gsn.zip -d data/
	rm data/data_gsn.zip

format:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

train-cls:
	@echo "Running Experiment 1: Classification Only..."
	$(PYTHON) -m src.train --lambda_cnt 0.0 --lambda_cls 1.0

train-reg:
	@echo "Running Experiment 2: Regression Only..."
	$(PYTHON) -m src.train --lambda_cnt 1.0 --lambda_cls 0.0

train-mtl:
	@echo "Running Experiment 3: Multitask Target..."
	$(PYTHON) -m src.train --lambda_cnt 2.0 --lambda_cls 1.0

train-all: train-cls train-reg train-mtl

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +