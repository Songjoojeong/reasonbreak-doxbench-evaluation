.PHONY: install test demo

install:
	pip install -e ".[data,dev]"

test:
	pytest -q

demo:
	python examples/run_offline_demo.py
