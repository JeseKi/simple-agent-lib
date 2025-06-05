"""
Agent Lib 安装配置

智能体框架库的安装脚本
"""

from setuptools import setup, find_packages

# 读取 README 文件作为长描述
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "智能体框架库 - 提供上下文记忆、工具调用和LLM交互功能"

# 读取依赖列表
try:
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
except FileNotFoundError:
    requirements = [
        "pydantic>=2.0.0",
        "openai>=1.0.0",
        "requests>=2.28.0",
        "typing-extensions>=4.0.0",
    ]

setup(
    name="agent-lib",
    version="1.2.0",
    author="Jese Ki",
    author_email="209490107@qq.com",  # 替换为你的邮箱
    description="智能体框架库 - 提供上下文记忆、工具调用和LLM交互功能",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/agent-lib",  # 替换为你的GitHub地址
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # 如果需要命令行工具，可以在这里添加
            # "agent-cli=agent_lib.cli:main",
        ],
    },
    keywords="ai, agent, llm, openai, context, memory, tools",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/agent-lib/issues",
        "Source": "https://github.com/yourusername/agent-lib",
        "Documentation": "https://github.com/yourusername/agent-lib/blob/main/CONTEXT_MEMORY_GUIDE.md",
    },
    include_package_data=True,
    zip_safe=False,
) 