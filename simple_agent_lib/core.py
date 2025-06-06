"""
自主智能体框架核心模块

该模块包含:
- LLM API交互逻辑
- 自主智能体执行引擎
- 工具执行管理

公开接口:
- LLMAPIClient: LLM API客户端
- Agent: 自主智能体类

内部方法:
- _StreamingToolCallAccumulator: 流式工具调用累加器
- LLMAPIClient的各种私有方法
- AutonomousAgent的工具执行方法
"""

import json
import inspect
from typing import List, Dict, Any, Callable, Optional, AsyncIterator

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
)
from .tools import get_tool_registry, get_tool_schemas
from .client import LLMAPIClient

from .logger_config import (
    log_tool_execution,
    log_agent_iteration,
    log_agent_completion,
    log_llm_interaction,
    get_logger,
    enable_logging,
)

logger = get_logger("智能体")


class Agent:
    """
    自主智能体

    负责管理LLM交互、工具执行和对话历史
    """

    def __init__(
        self,
        llm_api_client: LLMAPIClient,
        tools: Optional[List[Callable]] = None,
        system_prompt: Optional[str] = None,
        context: Optional[Context] = None,
        debug_mode: bool = False,
        log_level: str = "INFO",
    ):
        """
        初始化自主智能体

        Args:
            llm_api_client: LLM API客户端
            tools: 可用工具列表，如果为None则使用全局注册的工具
            system_prompt: 系统提示词，如果提供则自动添加到上下文
            context: 上下文管理器，如果为None则创建新的空上下文
            debug_mode: 是否启用调试模式（控制库的日志输出），默认False
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        # 设置日志状态
        enable_logging(debug_mode, log_level)
        self.llm_api_client = llm_api_client

        if tools is not None:
            # 使用提供的工具列表
            self.tools: Dict[str, Callable] = {
                func.__name__: func for func in tools
            }
            self.tool_schemas: List[Dict[str, Any]] = [
                getattr(func, "_tool_schema")
                for func in tools
                if hasattr(func, "_tool_schema")
            ]
        else:
            # 使用全局注册的工具
            self.tools = get_tool_registry()
            self.tool_schemas = get_tool_schemas()

        # 初始化上下文
        self.context = context or Context()

        # 如果提供了系统提示词，添加到上下文
        if system_prompt:
            self.context.add_system_message(system_prompt)

        # 保持向后兼容性：维护原始的message_history用于与现有LLM API交互
        self.message_history: List[Dict[str, Any]] = []

    def _add_to_history(self, message_dict: Dict[str, Any]):
        """添加消息到历史记录"""
        self.message_history.append(message_dict)

    def _sync_context_to_history(self):
        """将上下文同步到message_history（用于LLM API调用）"""
        self.message_history = self.context.to_openai_messages()

    def _add_user_message_to_context(self, content: str):
        """添加用户消息到上下文"""
        self.context.add_user_message(content)
        self._sync_context_to_history()

    def _add_assistant_response_to_context(
        self, content: Optional[str] = None, tool_calls: Optional[List[ToolCall]] = None
    ):
        """添加助手响应到上下文"""
        self.context.add_assistant_message(content=content, tool_calls=tool_calls)
        self._sync_context_to_history()

    def _add_tool_results_to_context(self, tool_results: List[ToolResult]):
        """添加工具结果到上下文"""
        for result in tool_results:
            # 智能序列化工具结果
            def serialize_result(result):
                """智能序列化结果，处理Pydantic模型和其他类型"""
                if hasattr(result, "model_dump"):
                    # Pydantic v2模型
                    return json.dumps(result.model_dump())
                elif hasattr(result, "dict"):
                    # Pydantic v1模型
                    return json.dumps(result.dict())
                else:
                    try:
                        return json.dumps(result)
                    except TypeError:
                        # 如果无法序列化，转换为字符串
                        return json.dumps(str(result))

            result_content = (
                serialize_result(result.result)
                if not result.is_error
                else json.dumps(
                    {"error": result.error_message, "details": str(result.result)}
                )
            )

            self.context.add_tool_result(
                tool_call_id=result.tool_call_id,
                tool_name=result.name,
                result=result_content,
            )
        self._sync_context_to_history()

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        执行工具调用

        Args:
            tool_call: 工具调用信息

        Returns:
            ToolResult: 工具执行结果
        """
        tool_name = tool_call.name
        tool_func = self.tools.get(tool_name)

        if not tool_func:
            return ToolResult(
                tool_call_id=tool_call.call_id,
                name=tool_name,
                result=f"错误: 工具 '{tool_name}' 未找到。",
                is_error=True,
                error_message=f"工具 '{tool_name}' 未找到。",
            )

        try:
            # 检查是否为异步函数
            if inspect.iscoroutinefunction(tool_func):
                result = await tool_func(**tool_call.args)
            else:
                result = tool_func(**tool_call.args)

            log_tool_execution(tool_name, tool_call.args, True, result=result)
            return ToolResult(
                tool_call_id=tool_call.call_id, name=tool_name, result=result
            )
        except Exception as e:
            # error_details = traceback.format_exc()
            log_tool_execution(tool_name, tool_call.args, False, error=str(e))
            return ToolResult(
                tool_call_id=tool_call.call_id,
                name=tool_name,
                result=str(e),
                is_error=True,
                error_message=str(e),
            )

    def reset_history(self):
        """重置对话历史（但保留系统消息）"""
        # 保存系统消息
        system_messages = [msg for msg in self.context.messages if msg.role == "system"]

        # 重置上下文并重新添加系统消息
        self.context.messages = system_messages

        # 同步到message_history
        self._sync_context_to_history()

    async def run(
        self, initial_prompt: str, stream_mode: bool = True, max_iterations: int = -1
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        运行自主智能体

        Args:
            initial_prompt: 初始提示
            stream_mode: 是否使用流式模式
            max_iterations: 最大迭代次数，-1表示不限制迭代次数

        Yields:
            AgentStreamEvent: Agent事件流
        """

        self.reset_history()
        self._add_user_message_to_context(initial_prompt)

        current_llm_api_id_for_turn: Optional[str] = None
        iteration = 0

        # 无限迭代或有限迭代
        while max_iterations == -1 or iteration < max_iterations:
            iteration += 1
            log_agent_iteration(
                iteration, max_iterations if max_iterations > 0 else None
            )

            pending_tool_calls_from_llm: List[ToolCall] = []
            llm_had_finish_reason: Optional[str] = None

            if stream_mode:
                log_llm_interaction("调用LLM (流式模式)")
                async for event in self.llm_api_client.chat_completion_stream(
                    self.message_history, self.tool_schemas
                ):
                    yield event
                    current_llm_api_id_for_turn = event.llm_response_id

                    if isinstance(event, ToolCallCompleteEvent):
                        pending_tool_calls_from_llm.append(event.tool_call)
                    elif isinstance(event, LLMEndReasonEvent):
                        llm_had_finish_reason = event.finish_reason
                        if event.finish_reason == "stop":
                            log_agent_completion(
                                "LLM决定停止",
                                iteration,
                                max_iterations if max_iterations > 0 else None,
                            )
                            return
                        elif event.finish_reason == "tool_calls":
                            logger.info(
                                f"LLM请求 {len(pending_tool_calls_from_llm)} 个工具"
                            )
            else:
                log_llm_interaction("调用LLM (非流式模式)")
                llm_output: LLMOutput = (
                    await self.llm_api_client.chat_completion_non_stream(
                        self.message_history, self.tool_schemas
                    )
                )
                current_llm_api_id_for_turn = llm_output.id
                llm_had_finish_reason = llm_output.finish_reason

                # 产生等效事件
                if llm_output.aggregated_reasoning_content:
                    yield ReasoningChunkEvent(
                        text=llm_output.aggregated_reasoning_content,
                        llm_response_id=current_llm_api_id_for_turn,
                    )
                if llm_output.aggregated_content:
                    yield ContentChunkEvent(
                        text=llm_output.aggregated_content,
                        llm_response_id=current_llm_api_id_for_turn,
                    )

                pending_tool_calls_from_llm.extend(llm_output.tool_calls)

                if llm_output.finish_reason:
                    yield LLMEndReasonEvent(
                        finish_reason=llm_output.finish_reason,
                        llm_response_id=current_llm_api_id_for_turn,
                    )

                if llm_output.finish_reason == "stop":
                    log_agent_completion(
                        "LLM决定停止",
                        iteration,
                        max_iterations if max_iterations > 0 else None,
                    )
                    return
                elif llm_output.finish_reason == "tool_calls":
                    logger.info(f"LLM请求 {len(pending_tool_calls_from_llm)} 个工具")

            if (
                not pending_tool_calls_from_llm
                and llm_had_finish_reason != "tool_calls"
            ):
                log_agent_completion(
                    "LLM未请求工具且未明确停止",
                    iteration,
                    max_iterations if max_iterations > 0 else None,
                )
                if llm_had_finish_reason and llm_had_finish_reason != "stop":
                    yield LLMEndReasonEvent(
                        finish_reason=llm_had_finish_reason,
                        llm_response_id=current_llm_api_id_for_turn,
                    )
                return

            if not pending_tool_calls_from_llm:
                log_agent_completion(
                    f"finish_reason为 '{llm_had_finish_reason}' 但无工具调用",
                    iteration,
                    max_iterations if max_iterations > 0 else None,
                )
                return

            # 将助手的工具调用请求添加到上下文
            if pending_tool_calls_from_llm:
                self._add_assistant_response_to_context(
                    tool_calls=pending_tool_calls_from_llm
                )

            # 执行工具
            executed_tool_results: List[ToolResult] = []
            logger.debug(f"  执行 {len(pending_tool_calls_from_llm)} 个工具...")
            for tool_to_call in pending_tool_calls_from_llm:
                logger.debug(f"    调用: {tool_to_call.name} 参数: {tool_to_call.args}")
                result = await self._execute_tool(tool_to_call)
                executed_tool_results.append(result)

            # 产生所有工具结果
            yield AllToolResultsEvent(
                results=executed_tool_results,
                llm_response_id=current_llm_api_id_for_turn,
            )

            # 将工具结果添加到上下文
            self._add_tool_results_to_context(executed_tool_results)

            pending_tool_calls_from_llm.clear()

            if iteration == max_iterations - 1:
                logger.info(f"[警告] Agent达到最大迭代次数 ({max_iterations})。")
