.PHONY: help install build test clean demo api

help:
	@echo "EDON CAV Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make install    - Install dependencies"
	@echo "  make build      - Build CAV dataset (10k samples)"
	@echo "  make test       - Run unit tests"
	@echo "  make clean      - Clean generated files"
	@echo "  make demo       - Build dataset and launch API"
	@echo "  make api        - Launch FastAPI server"

install:
	pip install -r requirements.txt

build:
	python cli.py build-cav --n 10000

test:
	pytest tests/ -v

clean:
	rm -rf data/*.json
	rm -rf models/*.joblib
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

demo: build api

api:
	cd api && python main.py

