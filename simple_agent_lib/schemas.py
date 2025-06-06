"""
自主智能体框架的数据模型定义模块

该模块包含:
- API请求/响应相关的底层模型
- Agent输出的标准化数据模型
- Agent事件流的事件类型定义

公开接口:
- APIMessage: API消息模型
- ToolCall: 工具调用模型
- ToolResult: 工具执行结果模型
- LLMOutput: LLM输出模型
- AgentStreamEvent: Agent事件流类型

内部方法:
- 各种Pydantic模型的验证器
"""

import json
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

from .exceptions import ContextTokenLimitExceededError

# --- API 请求/响应相关的底层模型 ---


class APIFunctionCall(BaseModel):
    """API函数调用模型"""

    name: Optional[str] = None  # 流式响应中可能为None
    arguments: Optional[str] = None  # 流式响应中可能为None，参数的原始JSON字符串


class APIToolCall(BaseModel):
    """API工具调用模型"""

    index: int
    id: Optional[str] = None  # 流式响应中可能为None
    type: Optional[Literal["function"]] = None  # 流式响应中可能为None
    function: APIFunctionCall


class APIMessage(BaseModel):
    """API消息模型"""

    role: str
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[APIToolCall]] = None


class APIChoiceDelta(BaseModel):
    """API选择增量模型（用于流式响应）"""

    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[APIToolCall]] = None


class APIChoice(BaseModel):
    """API选择模型"""

    index: int
    message: Optional[APIMessage] = None  # 非流式响应
    delta: Optional[APIChoiceDelta] = None  # 流式响应
    finish_reason: Optional[Literal["stop", "tool_calls", "length"]] = None


class APIUsage(BaseModel):
    """API使用情况模型"""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    completion_tokens_details: Optional[Dict[str, Any]] = None


class APIResponseNonStreamed(BaseModel):
    """非流式API响应模型"""

    id: str
    object: str  # "chat.completion"
    created: int
    model: str
    choices: List[APIChoice]
    usage: Optional[APIUsage] = None
    system_fingerprint: Optional[str] = None


# class APIResponseChunk(BaseModel):
#     """流式API响应块模型"""
# 
#     id: str
#     object: str  # "chat.completion.chunk"
#     created: int
#     model: str
#     choices: List[APIChoice]
#     system_fingerprint: Optional[str] = None
#     usage: Optional[APIUsage] = None


# --- Agent 输出的标准化数据模型 ---


class ToolCall(BaseModel):
    """表示LLM请求的完整工具调用，准备执行"""

    call_id: str = Field(alias="id")  # 匹配APIToolCall的id字段
    name: str
    args: Dict[str, Any]  # 解析后的参数
    index: int

    @field_validator("args", mode="before")
    @classmethod
    def parse_arguments_string(cls, v):
        """解析参数字符串为字典"""
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"参数JSON字符串无效: {e}")
        return v


class ToolResult(BaseModel):
    """工具执行结果"""

    tool_call_id: str  # 对应ToolCall.call_id
    name: str
    result: Any
    is_error: bool = False
    error_message: Optional[str] = None


class LLMOutput(BaseModel):
    """
    LLM交互的结构化输出
    可用于流式数据块和最终聚合数据
    """

    id: Optional[str] = None  # LLM响应的唯一ID

    # 流式时为单个块，聚合时为所有块的列表或最终值
    content_chunks: List[str] = Field(default_factory=list)
    reasoning_content_chunks: List[str] = Field(default_factory=list)

    # 流式时逐个产生，聚合时包含所有工具调用
    tool_calls: List[ToolCall] = Field(default_factory=list)

    finish_reason: Optional[Literal["stop", "tool_calls", "length"]] = None

    @property
    def aggregated_content(self) -> Optional[str]:
        """聚合的内容"""
        return "".join(self.content_chunks) if self.content_chunks else None

    @property
    def aggregated_reasoning_content(self) -> Optional[str]:
        """聚合的推理内容"""
        return (
            "".join(self.reasoning_content_chunks)
            if self.reasoning_content_chunks
            else None
        )


# --- Agent流式执行产生的事件 ---


class ReasoningChunkEvent(BaseModel):
    """推理块事件"""

    text: str
    llm_response_id: Optional[str] = None


class ContentChunkEvent(BaseModel):
    """内容块事件"""

    text: str
    llm_response_id: Optional[str] = None


class ToolCallCompleteEvent(BaseModel):
    """工具调用完成事件"""

    tool_call: ToolCall
    llm_response_id: Optional[str] = None


class AllToolResultsEvent(BaseModel):
    """所有工具结果事件"""

    results: List[ToolResult]
    llm_response_id: Optional[str] = None


class LLMEndReasonEvent(BaseModel):
    """LLM结束原因事件"""

    finish_reason: Literal["stop", "tool_calls", "length"]
    llm_response_id: Optional[str] = None


# Agent流事件联合类型
AgentStreamEvent = Union[
    ReasoningChunkEvent,
    ContentChunkEvent,
    ToolCallCompleteEvent,
    AllToolResultsEvent,
    LLMEndReasonEvent,
]

# --- 上下文记忆相关模型 ---


class ContextMessage(BaseModel):
    """上下文消息模型 - 表示对话历史中的单个消息"""

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None  # assistant的tool_calls时可能为空
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: str = Field(default_factory=lambda: f"msg_{datetime.now().timestamp()}")

    # 工具调用相关字段
    tool_calls: Optional[List[ToolCall]] = None  # assistant调用工具时
    tool_call_id: Optional[str] = None  # tool角色消息时的关联ID

    # 扩展字段
    metadata: Optional[Dict[str, Any]] = None


class Context(BaseModel):
    """上下文管理器 - 管理整个对话的消息历史"""

    messages: List[ContextMessage] = Field(default_factory=list)
    max_messages: Optional[int] = 20  # 最大消息数量
    max_tokens: Optional[int] = 64000  # 最大token数（粗略估算）

    def add_message(self, message: ContextMessage) -> None:
        """添加消息并自动管理上下文大小"""
        self.messages.append(message)
        self._cleanup_old_messages()

    def add_user_message(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """便捷方法：添加用户消息"""
        message = ContextMessage(role="user", content=content, metadata=metadata)
        self.add_message(message)

    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """便捷方法：添加助手消息"""
        message = ContextMessage(
            role="assistant", content=content, tool_calls=tool_calls, metadata=metadata
        )
        self.add_message(message)

    def add_system_message(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """便捷方法：添加系统消息"""
        message = ContextMessage(role="system", content=content, metadata=metadata)
        self.add_message(message)

    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """便捷方法：添加工具调用结果"""
        message = ContextMessage(
            role="tool",
            content=result,
            tool_call_id=tool_call_id,
            metadata={**{"tool_name": tool_name}, **(metadata or {})},
        )
        self.add_message(message)

    def to_openai_messages(self) -> List[Dict[str, Any]]:
        """转换为OpenAI API格式的消息列表"""
        openai_messages = []

        for msg in self.messages:
            openai_msg: dict = {
                "role": msg.role,
            }

            # 添加content（如果存在）
            if msg.content is not None:
                openai_msg["content"] = msg.content

            # 处理工具调用（assistant角色）
            if msg.tool_calls:
                openai_msg["tool_calls"] = []
                for tc in msg.tool_calls:
                    openai_msg["tool_calls"].append(
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.args),
                            },
                        }
                    )

            # 处理工具结果（tool角色）
            if msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id

            openai_messages.append(openai_msg)

        return openai_messages

    def get_recent_messages(self, count: int) -> List[ContextMessage]:
        """获取最近的N条消息"""
        return self.messages[-count:] if count > 0 else []

    def clear_old_messages(self) -> None:
        """手动清理超出限制的旧消息（保留系统消息）"""
        self._cleanup_old_messages()

    def estimate_tokens(self) -> int:
        """粗略估算当前上下文的token数量（基于字符数）"""
        total_chars = 0
        for msg in self.messages:
            if msg.content:
                total_chars += len(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total_chars += len(tc.name) + len(json.dumps(tc.args))

        # 粗略估算：1 token ≈ 4 个字符（针对中文和英文混合文本）
        return total_chars // 4

    def _cleanup_old_messages(self) -> None:
        """内部方法：清理超出限制的旧消息"""
        if not self.max_messages and not self.max_tokens:
            return

        # 分离不同类型的消息
        system_messages = [msg for msg in self.messages if msg.role == "system"]
        user_messages = [msg for msg in self.messages if msg.role == "user"]
        assistant_messages = [msg for msg in self.messages if msg.role == "assistant"]
        tool_messages = [msg for msg in self.messages if msg.role == "tool"]

        # 按时间排序非系统消息（保持对话逻辑）
        conversation_messages = user_messages + assistant_messages + tool_messages
        conversation_messages.sort(key=lambda x: x.timestamp)

        # 首先按消息数量限制（不包括系统消息）
        if self.max_messages and len(conversation_messages) > self.max_messages:
            # 保留最新的对话消息
            conversation_messages = conversation_messages[-(self.max_messages) :]

        # 然后按token数量限制
        if self.max_tokens:
            # 系统消息必须保留，先计算系统消息的tokens
            system_tokens = self._estimate_tokens_for_messages(system_messages)
            available_tokens = self.max_tokens - system_tokens

            # 如果系统消息本身就超过限制，抛出错误
            if system_tokens > self.max_tokens:
                raise ContextTokenLimitExceededError(
                    current_tokens=system_tokens,
                    token_limit=self.max_tokens,
                    system_tokens=system_tokens,
                )

            # 如果系统消息恰好等于限制，只保留系统消息
            if available_tokens <= 0:
                self.messages = system_messages
                return

            # 从最新消息开始，逐步添加，确保不超过token限制
            final_conversation_messages = []
            current_tokens = 0

            # 从最新消息往回添加
            for msg in reversed(conversation_messages):
                msg_tokens = self._estimate_tokens_for_messages([msg])
                if current_tokens + msg_tokens <= available_tokens:
                    final_conversation_messages.insert(0, msg)  # 插入到开头保持顺序
                    current_tokens += msg_tokens
                else:
                    # 如果当前消息会超出限制，停止添加更多消息
                    break

            # 检查是否至少能保留一条对话消息
            # 如果连最新的一条消息都无法保留，且这条消息是必要的（比如用户刚输入的），则抛出错误
            if not final_conversation_messages and conversation_messages:
                latest_msg = conversation_messages[-1]
                latest_msg_tokens = self._estimate_tokens_for_messages([latest_msg])
                minimum_required = system_tokens + latest_msg_tokens

                if minimum_required > self.max_tokens:
                    raise ContextTokenLimitExceededError(
                        current_tokens=minimum_required,
                        token_limit=self.max_tokens,
                        system_tokens=system_tokens,
                        minimum_required_tokens=minimum_required,
                    )

            conversation_messages = final_conversation_messages

        # 重新组合消息列表：系统消息 + 按时间排序的对话消息
        self.messages = system_messages + sorted(
            conversation_messages, key=lambda x: x.timestamp
        )

    def _estimate_tokens_for_messages(self, messages: List[ContextMessage]) -> int:
        """估算指定消息列表的token数量"""
        total_chars = 0
        for msg in messages:
            if msg.content:
                total_chars += len(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total_chars += len(tc.name) + len(json.dumps(tc.args))
        return total_chars // 4
