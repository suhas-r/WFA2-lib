#!/usr/bin/env python3
"""
Setup script for PyWFA2 - Python bindings for WFA2 library
"""

import os
import sys
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

# Get the directory containing this setup.py file
BINDINGS_DIR = Path(__file__).parent
WFA2_ROOT = BINDINGS_DIR.parent.parent

# WFA2 library paths
WFA2_INCLUDE_DIRS = [
    str(WFA2_ROOT),
    str(WFA2_ROOT / "wavefront"),
    str(WFA2_ROOT / "alignment"),
    str(WFA2_ROOT / "system"),
    str(WFA2_ROOT / "utils"),
    numpy.get_include()
]

WFA2_LIBRARY_DIRS = [
    str(WFA2_ROOT / "lib")
]

WFA2_LIBRARIES = ["wfa"]

# Compiler flags
EXTRA_COMPILE_ARGS = [
    "-O3",
    "-std=c99",
    "-Wall",
    "-Wno-unused-function",
    "-Wno-unused-variable",
    "-DWFA_PARALLEL",  # Enable parallel processing if available
]

EXTRA_LINK_ARGS = [
    "-fopenmp",  # Link OpenMP for parallel processing
]

# Check if WFA2 library exists
wfa_lib_path = WFA2_ROOT / "lib" / "libwfa.a"
if not wfa_lib_path.exists():
    print(f"Error: WFA2 library not found at {wfa_lib_path}")
    print("Please build the WFA2 library first by running 'make lib_wfa' in the WFA2-lib directory")
    sys.exit(1)

# Define the extension
extensions = [
    Extension(
        name="pywfa2",
        sources=["pywfa2.pyx"],
        include_dirs=WFA2_INCLUDE_DIRS,
        library_dirs=WFA2_LIBRARY_DIRS,
        libraries=WFA2_LIBRARIES,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        language="c"
    )
]

# Read long description from README if it exists
long_description = ""
readme_path = BINDINGS_DIR / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="pywfa2",
    version="2.3.0",
    author="WFA2 Python Bindings",
    author_email="your.email@example.com",
    description="High-performance Python bindings for the WFA2 sequence alignment library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smarco/WFA2-lib",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "embedsignature": True,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        }
    ),
    # Include type stub files (.pyi) for IDE IntelliSense support
    py_modules=["pywfa2"],
    data_files=[(".", ["pywfa2.pyi"])],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "cython>=0.29.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Cython",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="sequence alignment bioinformatics wavefront algorithm",
    zip_safe=False,
)
