"""
LLM API客户端

负责与OpenAI兼容的API进行交互，支持流式和非流式调用
"""

import os
import httpx
import json
from typing import List, Dict, Any, Optional, AsyncIterator
from pydantic import BaseModel

from .schemas import (
    APIResponseNonStreamed,
    ToolCall,
    LLMOutput,
    AgentStreamEvent,
    ReasoningChunkEvent,
    ContentChunkEvent,
    ToolCallCompleteEvent,
    LLMEndReasonEvent,
)

from .logger_config import (
    get_logger,
)

logger = get_logger("智能体")


class _StreamingToolCallAccumulator(BaseModel):
    """内部使用的流式工具调用累加器"""

    index: int
    id: str
    type: str
    name: str
    arguments_str: str = ""


class LLMAPIClient:
    """
    LLM API客户端

    负责与OpenAI兼容的API进行交互，支持流式和非流式调用
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        api_key: str = os.getenv("OPENAI_API_KEY", ""),
        httpx_client: Optional[httpx.AsyncClient] = None,
    ):
        """
        初始化LLM API客户端

        Args:
            model_name: 模型名称
            base_url: API基础URL
            api_key: API密钥
            httpx_client: 可选的httpx异步客户端
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self._client = httpx_client or httpx.AsyncClient()
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        流式聊天完成

        Args:
            messages: 消息历史
            tools: 可用工具列表

        Yields:
            AgentStreamEvent: Agent事件流
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        stream_headers = self._headers.copy()
        stream_headers["Accept"] = "text/event-stream"

        active_tool_accumulators: Dict[int, _StreamingToolCallAccumulator] = {}
        current_llm_api_id: Optional[str] = None

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=stream_headers,
                json=payload,
                timeout=30.0,
            ) as response:
                # 在开始处理流之前检查状态码
                if response.status_code >= 400:
                    # 读取错误响应内容
                    error_content = b""
                    async for chunk in response.aiter_bytes():
                        error_content += chunk

                    # 尝试解析错误信息
                    try:
                        error_text = error_content.decode("utf-8")
                        error_json = json.loads(error_text)
                        if isinstance(error_json, dict) and "error" in error_json:
                            error_info = error_json["error"]
                            if isinstance(error_info, dict) and "message" in error_info:
                                error_details = f"API错误: {error_info['message']}"
                            else:
                                error_details = f"API错误: {error_info}"
                        else:
                            error_details = f"响应内容: {error_text[:200]}..."
                    except Exception:
                        error_details = f"HTTP {response.status_code} 错误"

                    logger.error(
                        f"[错误] HTTP错误: {response.status_code} - {error_details}"
                    )
                    response.raise_for_status()

                # 如果状态码正常，继续处理流

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_json_str = line[len("data: ") :].strip()
                        if data_json_str == "[DONE]":
                            break
                        if not data_json_str:
                            continue

                        try:
                            # 直接解析原始JSON，不进行预过滤
                            import json as json_module

                            chunk_json = json_module.loads(data_json_str)
                            current_llm_api_id = chunk_json.get("id")

                            # 安全地获取choices和delta
                            choices = chunk_json.get("choices", [])
                            if not choices:
                                continue

                            choice = choices[0]
                            delta = choice.get("delta", {})
                            finish_reason = choice.get("finish_reason")

                            # 处理reasoning_content（推理模型的思考过程）
                            reasoning_content = delta.get("reasoning_content")
                            if reasoning_content:
                                yield ReasoningChunkEvent(
                                    text=reasoning_content,
                                    llm_response_id=current_llm_api_id,
                                )

                            # 处理普通content
                            content = delta.get("content")
                            if content:
                                yield ContentChunkEvent(
                                    text=content,
                                    llm_response_id=current_llm_api_id,
                                )

                            # 处理tool_calls
                            tool_calls = delta.get("tool_calls", [])
                            if tool_calls:
                                for tc_delta in tool_calls:
                                    idx = tc_delta.get("index", 0)
                                    tc_id = tc_delta.get("id")
                                    tc_type = tc_delta.get("type")
                                    function_info = tc_delta.get("function", {})
                                    function_name = function_info.get("name")
                                    function_args = function_info.get("arguments", "")

                                    # 如果这是一个新的工具调用开始（有完整的id、type、name）
                                    if tc_id and tc_type and function_name:
                                        # 先完成之前可能未完成的工具调用
                                        for prev_idx in list(
                                            active_tool_accumulators.keys()
                                        ):
                                            if prev_idx != idx:
                                                acc = active_tool_accumulators.pop(
                                                    prev_idx
                                                )
                                                try:
                                                    parsed_args = json.loads(
                                                        acc.arguments_str or "{}"
                                                    )
                                                    yield ToolCallCompleteEvent(
                                                        tool_call=ToolCall(
                                                            id=acc.id,
                                                            name=acc.name,
                                                            args=parsed_args,
                                                            index=acc.index,
                                                        ),
                                                        llm_response_id=current_llm_api_id,
                                                    )
                                                except json.JSONDecodeError as e:
                                                    logger.error(
                                                        f"[错误] 解析工具参数失败 {acc.name}: {e}"
                                                    )

                                        # 创建新的累加器
                                        active_tool_accumulators[idx] = (
                                            _StreamingToolCallAccumulator(
                                                index=idx,
                                                id=tc_id,
                                                type=tc_type,
                                                name=function_name,
                                                arguments_str=function_args,
                                            )
                                        )

                                    # 如果只是arguments的增量更新
                                    elif (
                                        function_args
                                        and idx in active_tool_accumulators
                                    ):
                                        active_tool_accumulators[
                                            idx
                                        ].arguments_str += function_args

                            # 处理finish_reason
                            if finish_reason:
                                # 完成所有剩余的工具调用
                                for (
                                    idx_rem,
                                    acc_rem,
                                ) in active_tool_accumulators.items():
                                    try:
                                        parsed_args_rem = json.loads(
                                            acc_rem.arguments_str or "{}"
                                        )
                                        yield ToolCallCompleteEvent(
                                            tool_call=ToolCall(
                                                id=acc_rem.id,
                                                name=acc_rem.name,
                                                args=parsed_args_rem,
                                                index=acc_rem.index,
                                            ),
                                            llm_response_id=current_llm_api_id,
                                        )
                                    except json.JSONDecodeError as e:
                                        logger.error(
                                            f"[错误] 解析工具参数失败 {acc_rem.name}: {e}"
                                        )

                                active_tool_accumulators.clear()
                                yield LLMEndReasonEvent(
                                    finish_reason=finish_reason,
                                    llm_response_id=current_llm_api_id,
                                )

                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"[警告] JSON解码失败: {e}, 行: '{data_json_str}'"
                            )
                        except Exception as e:
                            logger.error(
                                f"[错误] 处理SSE块失败: {e}, 行: '{data_json_str}'"
                            )

        except httpx.HTTPStatusError:
            # 错误已经在上面被记录了，这里只需要重新抛出
            raise
        except httpx.RequestError as e:
            logger.error(f"[错误] 请求错误: {e}")
            raise

    async def chat_completion_non_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMOutput:
        """
        非流式聊天完成

        Args:
            messages: 消息历史
            tools: 可用工具列表

        Returns:
            LLMOutput: LLM输出结果
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            response_data = APIResponseNonStreamed.model_validate(response.json())

            output = LLMOutput(id=response_data.id)
            if response_data.choices:
                choice = response_data.choices[0]
                output.finish_reason = choice.finish_reason
                if choice.message:
                    if choice.message.reasoning_content:
                        output.reasoning_content_chunks.append(
                            choice.message.reasoning_content
                        )
                    if choice.message.content:
                        output.content_chunks.append(choice.message.content)
                    if choice.message.tool_calls:
                        for tc_api in choice.message.tool_calls:
                            try:
                                parsed_args = json.loads(
                                    tc_api.function.arguments or "{}"
                                )
                                output.tool_calls.append(
                                    ToolCall(
                                        id=tc_api.id or "",
                                        name=tc_api.function.name or "",
                                        args=parsed_args,
                                        index=tc_api.index,
                                    )
                                )
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"[错误] 非流式模式解析工具参数失败 {tc_api.function.name}: {e}, 原始: '{tc_api.function.arguments}'"
                                )
            return output

        except httpx.HTTPStatusError as e:
            # 安全地获取错误详情
            try:
                error_text = e.response.text
                error_json = json.loads(error_text)
                if isinstance(error_json, dict) and "error" in error_json:
                    error_info = error_json["error"]
                    if isinstance(error_info, dict) and "message" in error_info:
                        error_details = f"API错误: {error_info['message']}"
                    else:
                        error_details = f"API错误: {error_info}"
                else:
                    error_details = f"响应内容: {error_text[:200]}..."
            except Exception:
                error_details = f"HTTP {e.response.status_code} 错误"

            logger.error(
                f"[错误] 非流式HTTP错误: {e.response.status_code} - {error_details}"
            )
            raise
        except httpx.RequestError as e:
            logger.error(f"[错误] 非流式请求错误: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"[错误] 非流式JSON解码失败: {e}")
            raise
