"""
Setup script for Patent/IP Intelligence System
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read long description from README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    for line in requirements_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="patent-intelligence",
    version="1.0.0",
    author="Patent Intelligence Team",
    author_email="team@example.com",
    description="Patent/IP Intelligence System for pharmaceutical patent cliff tracking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UMwai/patent-ip-intelligence",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.12.0",
            "isort>=5.13.0",
            "flake8>=6.1.0",
            "mypy>=1.8.0",
            "bandit>=1.7.6",
            "safety>=2.3.5",
        ],
        "test": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.12.0",
            "responses>=0.24.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "patent-intelligence=src.pipeline:main",
            "patent-backfill=scripts.backfill_historical_data:main",
        ],
    },
    package_data={
        "": ["*.yaml", "*.yml", "*.sql"],
    },
    include_package_data=True,
)
