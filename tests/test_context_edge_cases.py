#!/usr/bin/env python3
"""
上下文管理边界测试

测试各种边界情况下的上下文管理：
1. 纯文本输出（无工具调用）
2. 多轮工具调用和文本输出
3. 混合场景测试
"""

import asyncio
import os
import sys
from typing import Dict, Any
from pydantic import BaseModel

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_agent_lib import LLMAPIClient, Agent, Context, tool
from simple_agent_lib.schemas import (
    ReasoningChunkEvent,
    ContentChunkEvent,
    ToolCallCompleteEvent,
    AllToolResultsEvent,
    LLMEndReasonEvent,
)


# 测试用工具
class WeatherResponse(BaseModel):
    """天气响应模型"""
    city: str
    temperature: float
    weather: str


@tool
def get_weather(city: str) -> WeatherResponse:
    """获取指定城市的天气信息"""
    city_weather_map = {
        "北京": {"temperature": 25.0, "weather": "晴天"},
        "上海": {"temperature": 22.0, "weather": "多云"},
        "广州": {"temperature": 28.0, "weather": "小雨"},
        "深圳": {"temperature": 27.0, "weather": "阴天"},
    }
    
    weather_info = city_weather_map.get(city, {"temperature": 20.0, "weather": "晴天"})
    return WeatherResponse(
        city=city,
        temperature=weather_info["temperature"],
        weather=weather_info["weather"],
    )


@tool
def calculate_sum(a: float, b: float) -> Dict[str, Any]:
    """计算两个数的和"""
    result = a + b
    return {
        "a": a,
        "b": b,
        "sum": result,
        "operation": f"{a} + {b} = {result}"
    }


@tool
def get_time() -> Dict[str, str]:
    """获取当前时间（模拟）"""
    return {
        "current_time": "2025-07-01 12:00:00",
        "timezone": "UTC+8",
        "day_of_week": "星期一"
    }


async def create_test_agent() -> Agent:
    """创建测试用的Agent"""
    api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
    api_key = os.getenv("OPENAI_API_KEY", "test-key")
    model_name = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
    
    llm_client = LLMAPIClient(
        base_url=api_base,
        api_key=api_key,
        model_name=model_name
    )
    
    agent = Agent(
        llm_api_client=llm_client,
        tools=[get_weather, calculate_sum, get_time],
        system_prompt="你是一个测试助手。",
        debug_mode=True
    )
    
    return agent


async def test_pure_text_response(stream_mode: bool) -> list[dict]:
    """测试纯文本响应（无工具调用）"""
    mode_name = "流式" if stream_mode else "非流式"
    print(f"\n--- 测试纯文本响应（{mode_name}模式）---")
    
    agent = await create_test_agent()
    
    # 使用一个不会触发工具调用的简单问题
    prompt = "请用一句话介绍一下Python编程语言的特点。"
    print(f"用户输入: {prompt}")
    
    content_chunks = []
    async for event in agent.run(initial_prompt=prompt, stream_mode=stream_mode):
        if isinstance(event, ContentChunkEvent):
            content_chunks.append(event.text)
            print(f"内容: {event.text}", end="")
        elif isinstance(event, ToolCallCompleteEvent):
            print(f"\n⚠️  意外的工具调用: {event.tool_call.name}")
        elif isinstance(event, LLMEndReasonEvent):
            print(f"\n结束原因: {event.finish_reason}")
    
    print(f"\n收集到的内容块数量: {len(content_chunks)}")
    print(f"完整内容: {''.join(content_chunks)}")
    
    context_messages = agent.context.to_openai_messages()
    print(f"最终上下文消息数量: {len(context_messages)}")
    
    return context_messages


async def test_multiple_tools_and_text(stream_mode: bool) -> list[dict]:
    """测试多轮工具调用和文本输出"""
    mode_name = "流式" if stream_mode else "非流式"
    print(f"\n--- 测试多轮工具调用（{mode_name}模式）---")
    
    agent = await create_test_agent()
    
    # 使用一个可能触发多次工具调用的复杂问题
    prompt = "请告诉我北京和上海的天气，然后计算两个城市温度的和，最后告诉我现在的时间。请详细说明每一步的结果。"
    print(f"用户输入: {prompt}")
    
    tool_calls_count = 0
    content_chunks = []
    
    async for event in agent.run(initial_prompt=prompt, stream_mode=stream_mode):
        if isinstance(event, ContentChunkEvent):
            content_chunks.append(event.text)
            print(f"内容: {event.text}", end="")
        elif isinstance(event, ToolCallCompleteEvent):
            tool_calls_count += 1
            print(f"\n🔧 工具调用 #{tool_calls_count}: {event.tool_call.name}({event.tool_call.args})")
        elif isinstance(event, AllToolResultsEvent):
            print(f"✅ 工具执行完成，结果数量: {len(event.results)}")
        elif isinstance(event, LLMEndReasonEvent):
            print(f"\n🏁 结束原因: {event.finish_reason}")
    
    print(f"\n总工具调用次数: {tool_calls_count}")
    print(f"收集到的内容块数量: {len(content_chunks)}")
    
    context_messages = agent.context.to_openai_messages()
    print(f"最终上下文消息数量: {len(context_messages)}")
    
    return context_messages


def analyze_context_completeness(context_messages: list[dict], test_name: str) -> bool:
    """分析上下文的完整性"""
    print(f"\n=== {test_name} 上下文分析 ===")
    
    assistant_messages = [msg for msg in context_messages if msg.get('role') == 'assistant']
    tool_messages = [msg for msg in context_messages if msg.get('role') == 'tool']
    
    print(f"助手消息数量: {len(assistant_messages)}")
    print(f"工具消息数量: {len(tool_messages)}")
    
    # 检查是否有助手文本内容
    text_responses = [msg for msg in assistant_messages if msg.get('content')]
    tool_call_requests = [msg for msg in assistant_messages if msg.get('tool_calls')]
    
    print(f"包含文本内容的助手消息: {len(text_responses)}")
    print(f"包含工具调用的助手消息: {len(tool_call_requests)}")
    
    # 详细列出所有消息
    for i, msg in enumerate(context_messages, 1):
        role = msg.get('role', '未知')
        content = (msg.get('content', '') or '')[:50] + "..." if len((msg.get('content', '') or '')) > 50 else (msg.get('content', '') or '')
        tool_calls_count = len(msg.get('tool_calls', []))
        tool_call_id = msg.get('tool_call_id', '')
        
        print(f"  {i}. {role}: '{content}' (工具调用: {tool_calls_count}) (tool_call_id: {tool_call_id})")
    
    return True


async def test_context_edge_cases():
    """主要边界测试函数"""
    print("🧪 开始上下文管理边界测试...")
    
    results = []
    
    # 测试1: 纯文本响应 - 流式
    try:
        stream_context = await test_pure_text_response(stream_mode=True)
        analyze_context_completeness(stream_context, "纯文本响应（流式）")
        results.append(("纯文本响应（流式）", True, len(stream_context)))
    except Exception as e:
        print(f"❌ 纯文本响应（流式）测试失败: {e}")
        results.append(("纯文本响应（流式）", False, 0))
    
    # 测试2: 纯文本响应 - 非流式
    try:
        non_stream_context = await test_pure_text_response(stream_mode=False)
        analyze_context_completeness(non_stream_context, "纯文本响应（非流式）")
        results.append(("纯文本响应（非流式）", True, len(non_stream_context)))
    except Exception as e:
        print(f"❌ 纯文本响应（非流式）测试失败: {e}")
        results.append(("纯文本响应（非流式）", False, 0))
    
    # 测试3: 多轮工具调用 - 流式
    try:
        multi_stream_context = await test_multiple_tools_and_text(stream_mode=True)
        analyze_context_completeness(multi_stream_context, "多轮工具调用（流式）")
        results.append(("多轮工具调用（流式）", True, len(multi_stream_context)))
    except Exception as e:
        print(f"❌ 多轮工具调用（流式）测试失败: {e}")
        results.append(("多轮工具调用（流式）", False, 0))
    
    # 测试4: 多轮工具调用 - 非流式
    try:
        multi_non_stream_context = await test_multiple_tools_and_text(stream_mode=False)
        analyze_context_completeness(multi_non_stream_context, "多轮工具调用（非流式）")
        results.append(("多轮工具调用（非流式）", True, len(multi_non_stream_context)))
    except Exception as e:
        print(f"❌ 多轮工具调用（非流式）测试失败: {e}")
        results.append(("多轮工具调用（非流式）", False, 0))
    
    # 汇总结果
    print(f"\n📊 边界测试结果汇总:")
    passed = 0
    for test_name, success, msg_count in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status} (消息数: {msg_count})")
        if success:
            passed += 1
    
    print(f"\n总体结果: {passed}/{len(results)} 通过")
    
    # 检查关键期望
    print(f"\n🔍 关键检查:")
    
    # 1. 纯文本模式下应该有助手回复
    pure_text_tests = [r for r in results if "纯文本" in r[0] and r[1]]
    if pure_text_tests:
        print(f"✅ 纯文本响应测试: {len(pure_text_tests)}/2 通过")
        for name, _, count in pure_text_tests:
            if count >= 2:  # 至少系统消息 + 用户消息 + 助手回复
                print(f"  ✅ {name}: 上下文包含助手回复 (消息数: {count})")
            else:
                print(f"  ⚠️  {name}: 上下文可能缺少助手回复 (消息数: {count})")
    
    # 2. 多轮工具调用应该有完整的对话流
    multi_tool_tests = [r for r in results if "多轮" in r[0] and r[1]]
    if multi_tool_tests:
        print(f"✅ 多轮工具调用测试: {len(multi_tool_tests)}/2 通过")
        for name, _, count in multi_tool_tests:
            if count >= 6:  # 系统+用户+多个助手+工具消息
                print(f"  ✅ {name}: 上下文包含完整对话流 (消息数: {count})")
            else:
                print(f"  ⚠️  {name}: 上下文可能不完整 (消息数: {count})")
    
    return passed == len(results)


def main():
    """主函数"""
    return asyncio.run(test_context_edge_cases())


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 