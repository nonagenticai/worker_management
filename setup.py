"""Legacy setuptools entrypoint — actual build is driven by pyproject.toml + hatchling."""

from setuptools import find_packages, setup

setup(
    name="worker_management",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "pydantic-ai",
        "httpx",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-asyncio",
            "pytest-mock",
            "pytest-cov",
            "flake8",
            "isort",
            "pylint",
            "black",
            "bandit",
            "coverage",
            "vulture",
            "xenon",
            "pydocstyle",
            "mkdocs",
            "Sphinx",
        ],
    },
    entry_points={
        "console_scripts": [
            "worker-management=src.api.worker_management_api:app",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
