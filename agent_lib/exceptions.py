"""
智能体框架的自定义异常

该模块包含:
- 上下文管理相关的异常
- 工具执行相关的异常
- LLM交互相关的异常

公开接口:
- ContextTokenLimitExceededError: 上下文token限制超出错误
- ContextManagementError: 上下文管理基础错误类
"""

from typing import Optional


class ContextManagementError(Exception):
    """上下文管理基础错误类"""

    def __init__(self, message: str, current_tokens: int = 0, token_limit: int = 0):
        self.current_tokens = current_tokens
        self.token_limit = token_limit
        super().__init__(message)


class ContextTokenLimitExceededError(ContextManagementError):
    """
    上下文token限制超出错误

    当即使删除所有可删除的消息后，仍然超出token限制时抛出此错误。
    通常发生在：
    1. 系统消息本身就超出了token限制
    2. 必要的消息（如系统消息 + 最新用户消息）超出了限制
    """

    def __init__(
        self,
        current_tokens: int,
        token_limit: int,
        system_tokens: int = 0,
        minimum_required_tokens: int = 0,
        message: Optional[str] = None,
    ):
        self.system_tokens = system_tokens
        self.minimum_required_tokens = minimum_required_tokens

        if message is None:
            if system_tokens >= token_limit:
                message = (
                    f"系统消息本身使用了 {system_tokens} tokens，"
                    f"超出了设置的限制 {token_limit} tokens。"
                    f"请减少系统消息的长度或增加token限制。"
                )
            else:
                message = (
                    f"即使删除所有可删除的消息，当前所需的最少 {minimum_required_tokens} tokens "
                    f"仍然超出了设置的限制 {token_limit} tokens。"
                    f"请增加token限制或减少必要消息的长度。"
                )

        super().__init__(message, current_tokens, token_limit)


class ContextMessageError(ContextManagementError):
    """上下文消息相关错误"""

    pass


class ToolExecutionError(Exception):
    """工具执行相关错误"""

    def __init__(
        self, tool_name: str, message: str, original_error: Optional[Exception] = None
    ):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"工具 '{tool_name}' 执行失败: {message}")


class LLMInteractionError(Exception):
    """LLM交互相关错误"""

    pass
