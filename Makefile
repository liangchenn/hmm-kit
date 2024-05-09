# target
TARGET ?= main

# virtual env folder
VENV_NAME ?= .venv
PYTHON = ${VENV_NAME}/bin/python


.PHONY: help
help:
	@echo "Usage:"
	@echo "---------------------------------------------------------------------"
	@echo "  make help              : display this help message"
	@echo "  make all               : create virtual environment, activate it, install requirements.txt,"
	@echo "                           and install pre-commit hooks"
	@echo "  make env               : create virtual environment and activate it"
	@echo "  make create_venv       : create virtual environment"
	@echo "  make activate          : activate virtual environment"
	@echo "  make install           : install packages from requirements.txt"
	@echo "  make install-dev       : install packages from requirements-dev.txt"
	@echo "  make precommit         : install pre-commit hooks"
	@echo "  make tests             : run tests"
	@echo "  make clean             : clean up .ipynb_checkpoints and __pycache__"
	@echo "  make remove_env        : remove virtual environment"
	@echo "  make requirements      : export packages from requirements.txt and requirements-dev.txt"
	@echo "  make requirements-main : export packages from requirements.txt"
	@echo "  make requirements-dev  : export packages from requirements-dev.txt"
	@echo "---------------------------------------------------------------------"
	@echo ""


.PHONY: all
all: create_venv activate install-dev precommit


.PHONY: env
env: create_venv activate


.PHONY: create_venv
create_venv:
	@echo "creating virtual environment in ${VENV_NAME}/..."
	python -m venv ${VENV_NAME}


.PHONY: activate
activate:
	@echo "Run 'source $(VENV_NAME)/bin/activate' to activate the virtual environment."
	@echo "To exit the venv, please run 'deactivate'."


.PHONY: install
install:
	@if [ "$(DEP)" = "dev" ]; then \
		make install-dev; \
	else \
		make install-main; \
	fi


.PHONY: install-main
install-main:
	@echo "installing packages from requirements.txt"
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements.txt


.PHONY: install-dev
install-dev:
	@echo "installing packages from requirements-dev.txt"
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r requirements-dev.txt


.PHONY: precommit
precommit:
	@echo "installing pre-commit"
	$(VENV_NAME)/bin/pre-commit install


.PHONY: tests
tests:
	$(PYTHON) -m pytest tests/


.PHONY: remove_env
remove_env:
	@echo "removing virtual environment..."
	rm -rf ${VENV_NAME}


.PHONY: requirements
requirements: requirements-main requirements-dev


.PHONY: requirements-main
requirements-main:
	poetry export -f requirements.txt --output requirements.txt --without-hashes


.PHONY: requirements-dev
requirements-dev:
	poetry export -f requirements.txt --output requirements-dev.txt --dev --without-hashes


.PHONY: clean
clean:
	@echo "Cleaning up .ipynb_checkpoints and __pycache__..."
	@find . -name .ipynb_checkpoints -type d -prune -exec rm -rf {} +
	@find . -name __pycache__ -type d -prune -exec rm -rf {} +
	@find . -name *_cache -type d -prune -exec rm -rf {} +
	@echo "Cleanup done."
