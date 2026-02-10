"""Setup script for interpreter-agent-eval."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="interpreter-agent-eval",
    version="0.1.0",
    author="Faiz Ghifari",
    description="Evaluation framework for interpreter/translator agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/faizghifari/interpreter-agent-eval",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies will be optional - users install what they need
    ],
    extras_require={
        "google": ["google-genai>=0.1.0"],
        "openai": ["openai>=1.0.0"],
        "all": [
            "google-genai>=0.1.0",
            "openai>=1.0.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
)
