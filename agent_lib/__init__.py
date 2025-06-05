"""
Agent Lib - 智能体框架库

提供上下文记忆、工具调用和LLM交互功能的Python库。

主要功能:
- 🧠 智能上下文记忆 - 自动管理对话历史，支持token和消息数量限制
- 🛠️ 工具调用系统 - 装饰器式工具定义，自动参数验证
- 🤖 LLM交互框架 - 统一的OpenAI API接口
- ⚡ 异常处理 - 智能的错误处理和恢复策略

使用示例:
    from agent_lib import AutonomousAgent, Context, tool

    @tool
    def get_weather(city: str) -> str:
        return f"{city}今天晴朗，温度25°C"

    context = Context(max_tokens=4000, max_messages=50)
    agent = AutonomousAgent(context=context, system_prompt="你是一个智能助手")
    response = agent.run("北京天气怎么样？")
"""

__version__ = "1.2.0"
__author__ = "Jese Ki"
__email__ = "209490107@qq.com"
__description__ = "智能体框架库 - 提供上下文记忆、工具调用和LLM交互功能"

from .schemas import (
    ToolCall,
    ToolResult,
    LLMOutput,
    AgentStreamEvent,
    ReasoningChunkEvent,
    ContentChunkEvent,
    ToolCallCompleteEvent,
    AllToolResultsEvent,
    LLMEndReasonEvent,
    Context,
    ContextMessage,
)

from .tools import tool, get_tool_schemas, get_tool_registry, clear_tools

from .core import LLMAPIClient, AutonomousAgent

from .exceptions import (
    ContextTokenLimitExceededError,
    ContextManagementError,
    ToolExecutionError,
    LLMInteractionError,
)

__all__ = [
    # 核心类
    "LLMAPIClient",
    "AutonomousAgent",
    # 工具相关
    "tool",
    "get_tool_schemas",
    "get_tool_registry",
    "clear_tools",
    # 类型定义
    "ToolCall",
    "ToolResult",
    "LLMOutput",
    "AgentStreamEvent",
    "ReasoningChunkEvent",
    "ContentChunkEvent",
    "ToolCallCompleteEvent",
    "AllToolResultsEvent",
    "LLMEndReasonEvent",
    # 上下文记忆
    "Context",
    "ContextMessage",
    # 异常类
    "ContextTokenLimitExceededError",
    "ContextManagementError",
    "ToolExecutionError",
    "LLMInteractionError",
]
