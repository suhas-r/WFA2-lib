# Makefile for PyWFA2 Python bindings

.PHONY: build install clean test example check-wfa2

# Default target
all: check-wfa2 build

# Check if WFA2 library is built
check-wfa2:
	@echo "Checking WFA2 library..."
	@if [ ! -f ../../lib/libwfa.a ]; then \
		echo "Error: WFA2 library not found. Please run 'make lib_wfa' in the WFA2-lib root directory first."; \
		exit 1; \
	fi
	@echo "WFA2 library found ✓"

# Build the Python extension
build: check-wfa2
	@echo "Building PyWFA2..."
	python setup.py build_ext --inplace
	@echo "Build complete ✓"

# Install the package
install: check-wfa2
	@echo "Installing PyWFA2..."
	pip install -e .
	@echo "Installation complete ✓"

# Install in development mode with dependencies
install-dev: check-wfa2
	@echo "Installing PyWFA2 in development mode..."
	pip install -e ".[dev]"
	@echo "Development installation complete ✓"

# Run the example
example: 
	@echo "Running example..."
	python example.py

# Run tests (if test suite exists)
test:
	@echo "Running tests..."
	@if [ -f test_pywfa2.py ]; then \
		python test_pywfa2.py; \
	else \
		echo "No tests found. Running example instead..."; \
		python example.py; \
	fi

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -f *.so
	rm -f *.c
	rm -f pywfa2.html
	@echo "Clean complete ✓"

# Help
help:
	@echo "PyWFA2 Python Bindings Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Check WFA2 library and build (default)"
	@echo "  build      - Build the Python extension"
	@echo "  install    - Install the package with pip"
	@echo "  install-dev- Install in development mode"
	@echo "  example    - Run the example script"
	@echo "  test       - Run tests (or example if no tests)"
	@echo "  clean      - Remove build artifacts"
	@echo "  check-wfa2 - Check if WFA2 library is built"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Prerequisites:"
	@echo "  1. Build WFA2 library: cd ../.. && make lib_wfa"
	@echo "  2. Install dependencies: pip install cython numpy"
