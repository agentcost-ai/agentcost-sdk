"""
AgentCost SDK Setup

Install with: pip install -e .
"""

from pathlib import Path
from setuptools import setup, find_packages


def get_version() -> str:
    """Read version from VERSION file (single source of truth)."""
    version_file = Path(__file__).parent.parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return "0.1.0"


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agentcost",
    version=get_version(),
    author="Kushagra Agrawal",
    author_email="kushagraagrawal128@gmail.com",
    description="Track LLM costs in LangChain applications with zero code changes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agentcost-ai/agentcost-sdk",
    project_urls={
        "Bug Tracker": "https://github.com/agentcost-ai/agentcost-sdk/issues",
        "Documentation": "https://www.agentcost.tech/docs/sdk",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "tiktoken>=0.5.0",
        "requests>=2.28.0",
        "langchain-core>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
        "langchain": [
            "langchain>=0.1.0",
            "langchain-openai>=0.0.5",
            "langchain-anthropic>=0.1.0",
        ],
    },
    keywords=[
        "llm",
        "langchain",
        "openai",
        "anthropic",
        "cost-tracking",
        "tokens",
        "ai",
        "monitoring",
        "observability",
        "langchain ai agents cost tracking monitoring"
    ],
    include_package_data=True,
)

