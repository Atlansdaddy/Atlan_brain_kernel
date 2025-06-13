"""
Setup script for Atlan Brain Kernel
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="atlan-brain-kernel",
    version="0.1.0",
    author="Johnathan & Claude",
    description="A field-based resonance cognitive architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/atlan-brain-kernel",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "visualization": [
            "plotly>=5.0",
            "networkx>=2.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "atlan-demo=atlan_brain_kernel:run_basic_demo",
            "atlan-advanced=atlan_brain_kernel:run_advanced_demo",
        ],
    },
)