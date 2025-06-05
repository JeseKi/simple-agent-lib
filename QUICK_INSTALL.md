# Agent Lib 快速安装指南

## 🚀 三种安装方式

### 1. 本地开发安装（推荐用于开发）

```bash
# 进入项目目录
cd /path/to/agno

# 开发模式安装
pip install -e .

# 安装开发依赖
pip install -e ".[dev]"
```

### 2. 从构建的包安装

```bash
# 构建包
python -m build

# 安装wheel包
pip install dist/agent_lib-1.2.0-py3-none-any.whl

# 或安装源码包
pip install dist/agent_lib-1.2.0.tar.gz
```

### 3. 未来从PyPI安装

```bash
# 当发布到PyPI后
pip install agent-lib
```

## ✅ 验证安装

```bash
# 运行验证脚本
python test_installation.py
```

或者手动验证：

```python
import agent_lib
print(f"版本: {agent_lib.__version__}")

from agent_lib import Context, AutonomousAgent, tool
print("✅ 安装成功！")
```

## 🎯 快速开始

```python
from agent_lib import AutonomousAgent, Context, tool

# 定义工具
@tool
def calculate(a: float, b: float, op: str) -> float:
    """简单计算器"""
    if op == "+":
        return a + b
    elif op == "*":
        return a * b
    return 0

# 创建上下文
context = Context(max_tokens=4000, max_messages=50)

# 创建智能体
agent = AutonomousAgent(
    context=context,
    system_prompt="你是一个数学助手"
)

# 使用（需要配置LLM API）
# response = agent.run("计算 3 + 5")
```

## 📋 系统要求

- **Python**: >= 3.8
- **依赖**: pydantic, openai, requests, typing-extensions

## 🔗 更多信息

- [详细安装指南](INSTALL_GUIDE.md)
- [完整使用文档](CONTEXT_MEMORY_GUIDE.md)
- [API参考](README.md) 