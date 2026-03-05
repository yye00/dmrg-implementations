"""Setup script for PDMRG package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pdmrg",
    version="0.1.0",
    author="PDMRG Contributors",
    description="Parallel Density Matrix Renormalization Group",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/pdmrg",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "quimb>=1.4",
        "mpi4py>=3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-mpi",
        ],
    },
    entry_points={
        "console_scripts": [
            "pdmrg=pdmrg.dmrg:main",
        ],
    },
)
