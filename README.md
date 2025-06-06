# Agent Lib

**智能体框架库** - 提供上下文记忆、工具调用和LLM交互功能

## ✨ 功能特性

- 🧠 **智能上下文记忆** - 自动管理对话历史，支持token和消息数量限制
- 🛠️ **工具调用系统** - 装饰器式工具定义，自动参数验证
- 🤖 **LLM交互框架** - 统一的OpenAI API接口
- ⚡ **异常处理** - 智能的错误处理和恢复策略
- 📝 **类型安全** - 完整的类型提示支持

## 🚀 快速开始

### 安装

```bash
pip install simple-agent-lib
```

### 基础使用

```python
from simple_agent_lib import Agent, Context, tool

# 定义工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"{city}今天晴朗，温度25°C"

# 创建上下文
context = Context(max_tokens=4000, max_messages=50)

# 创建智能体
agent = Agent(
    context=context,
    system_prompt="你是一个智能助手"
)

# 运行对话
response = agent.run("北京天气怎么样？")
print(response)
```

### 上下文记忆

```python
from simple_agent_lib import Context

# 创建带限制的上下文
context = Context(max_tokens=1000, max_messages=20)

# 添加对话
context.add_user_message("你好")
context.add_assistant_message("你好！有什么可以帮助您的吗？")
context.add_user_message("请介绍一下Python")

# 获取OpenAI格式的消息
messages = context.to_openai_format()

# 自动清理旧消息，保持在限制范围内
print(f"当前消息数: {len(context.messages)}")
print(f"估算tokens: {context.estimate_tokens()}")
```

### 异常处理

```python
from simple_agent_lib import Context, ContextTokenLimitExceededError

try:
    context = Context(max_tokens=50)
    context.add_system_message("很长很长的系统消息..." * 100)
except ContextTokenLimitExceededError as e:
    print(f"Token限制超出: {e}")
    print(f"建议: 减少消息长度或增加token限制")
```

## 📚 详细文档

- [上下文记忆指南](CONTEXT_MEMORY_GUIDE.md) - 完整的功能文档和最佳实践
- [API参考](docs/api.md) - 详细的API文档
- [示例项目](examples/) - 实际使用示例

## 🛠️ 开发

### 安装开发依赖

```bash
pip install -e ".[dev]"
```

### 运行测试

```bash
pytest
```

### 代码格式化

```bash
black simple_agent_lib/
isort simple_agent_lib/
```

## 📋 需求

- Python >= 3.8
- pydantic >= 2.0.0
- openai >= 1.0.0

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交 issue 和 pull request！

## 🔗 相关链接

- [GitHub 仓库](https://github.com/yourusername/simple-agent-lib)
- [问题反馈](https://github.com/yourusername/simple-agent-lib/issues)
- [更新日志](CHANGELOG.md) 