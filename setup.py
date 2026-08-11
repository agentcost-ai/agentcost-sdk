"""
AgentCost SDK Setup

Install with: pip install -e .
"""

from pathlib import Path
from setuptools import setup, find_packages


def get_version() -> str:
    """Read version from VERSION file (single source of truth)."""
    version_file = Path(__file__).parent / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return "0.1.1"


with open(Path(__file__).parent / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agentcost",
    version=get_version(),
    author="Kushagra Agrawal",
    author_email="kushagraagrawal128@gmail.com",
    description="Track LLM costs across OpenAI, Anthropic, Gemini, LangChain, and 3500+ models with zero code changes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://agentcost.tech",
    project_urls={
        "Homepage": "https://agentcost.tech",
        "Documentation": "https://agentcost.tech/docs/sdk",
        "Source": "https://github.com/agentcost-ai/agentcost-sdk",
        "Bug Tracker": "https://github.com/agentcost-ai/agentcost-sdk/issues",
    },
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # Excluded: find_packages() otherwise ships "tests" to PyPI as a
    # top-level module, where it collides with any other package doing
    # the same.
    packages=find_packages(exclude=["tests", "tests.*"]),
    entry_points={
        "console_scripts": [
            "agentcost=agentcost.cli:main",
        ],
    },
    # 3.9 is the real floor: PEP 585 builtin generics (list[str], tuple[int,
    # int]) are used in runtime-evaluated annotations, which 3.8 cannot parse.
    python_requires=">=3.9",
    install_requires=[
        # 0.7.0 is the first release shipping the o200k_base encoding that
        # token_counter requests for the gpt-4o / gpt-4.1 / gpt-5 / o-series
        # families. On older versions that lookup raises and silently falls
        # back to cl100k_base, undercounting those models.
        "tiktoken>=0.7.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "openai": [
            "openai>=1.0.0",
        ],
        "anthropic": [
            "anthropic>=0.18.0",
        ],
        "gemini": [
            "google-genai>=1.0.0",
        ],
        "langchain": [
            "langchain-core>=0.1.0",
        ],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "google-genai>=1.0.0",
            "langchain-core>=0.1.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "mypy>=1.0.0",
            "ruff>=0.1.0",
        ],
    },
    keywords=[
        "llm",
        "openai",
        "anthropic",
        "gemini",
        "langchain",
        "langgraph",
        "ai-agents",
        "cost-tracking",
        "tokens",
        "observability",
        "monitoring",
        "llm-cost",
    ],
    include_package_data=True,
)

