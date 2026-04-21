.PHONY: install run test all

install:
	pip install -r requirements.txt

run:
	python main.py

test:
	pytest tests/ -v

all: install test run