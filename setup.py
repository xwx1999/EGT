from setuptools import setup, find_packages

setup(
    name="egt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.2",
        "torch>=1.8.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "lightgbm>=3.2.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "statsmodels>=0.13.2",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Enhanced Genomic Transformer for Animal Quantitative Trait Prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/EGT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.7",
) 