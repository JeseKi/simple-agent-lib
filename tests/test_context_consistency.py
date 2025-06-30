#!/usr/bin/env python3
"""
上下文一致性测试

测试流式模式和非流式模式下Agent的上下文管理是否一致
"""

import asyncio
import os
import sys
import random
from typing import Dict, Any
from pydantic import BaseModel
import pytest

# 添加项目根目录到Python路径，以使用当前项目的代码而不是已安装的版本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_agent_lib import LLMAPIClient, Agent, Context, tool
from simple_agent_lib.schemas import (
    ReasoningChunkEvent,
    ContentChunkEvent,
    ToolCallCompleteEvent,
    AllToolResultsEvent,
    LLMEndReasonEvent,
)


# 测试用的简单工具
class WeatherResponse(BaseModel):
    """天气响应模型"""
    city: str
    temperature: float
    weather: str


@tool
def get_weather(city: str) -> WeatherResponse:
    """获取指定城市的天气信息
    
    Args:
        city: 城市名称
        
    Returns:
        WeatherResponse: 天气信息
    """
    # 模拟获取天气信息，为了测试一致性使用固定值
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
    """计算两个数的和
    
    Args:
        a: 第一个数
        b: 第二个数
        
    Returns:
        Dict[str, Any]: 计算结果
    """
    result = a + b
    return {
        "a": a,
        "b": b,
        "sum": result,
        "operation": f"{a} + {b} = {result}"
    }


async def create_test_agent() -> Agent:
    """创建测试用的Agent"""
    # 使用环境变量或默认值
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
        tools=[get_weather, calculate_sum],
        system_prompt="你是一个测试助手，帮助用户进行测试。",
        debug_mode=True
    )
    
    return agent


async def run_conversation_stream(agent: Agent, prompts: list[str]) -> list[dict]:
    """运行流式对话并收集上下文"""
    collected_events = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- 流式模式 - 第{i+1}轮对话 ---")
        print(f"用户输入: {prompt}")
        
        # 第一轮使用run，后续使用continue_run
        if i == 0:
            async for event in agent.run(initial_prompt=prompt, stream_mode=True):
                collected_events.append(event)
                if isinstance(event, ContentChunkEvent):
                    print(f"内容: {event.text}", end="")
                elif isinstance(event, ToolCallCompleteEvent):
                    print(f"\n工具调用: {event.tool_call.name}({event.tool_call.args})")
                elif isinstance(event, AllToolResultsEvent):
                    print(f"工具结果数量: {len(event.results)}")
        else:
            async for event in agent.continue_run(prompt=prompt, stream_mode=True):
                collected_events.append(event)
                if isinstance(event, ContentChunkEvent):
                    print(f"内容: {event.text}", end="")
                elif isinstance(event, ToolCallCompleteEvent):
                    print(f"\n工具调用: {event.tool_call.name}({event.tool_call.args})")
                elif isinstance(event, AllToolResultsEvent):
                    print(f"工具结果数量: {len(event.results)}")
    
    # 返回最终的上下文消息
    return agent.context.to_openai_messages()


async def run_conversation_non_stream(agent: Agent, prompts: list[str]) -> list[dict]:
    """运行非流式对话并收集上下文"""
    
    for i, prompt in enumerate(prompts):
        print(f"\n--- 非流式模式 - 第{i+1}轮对话 ---")
        print(f"用户输入: {prompt}")
        
        # 第一轮使用run，后续使用continue_run
        if i == 0:
            async for event in agent.run(initial_prompt=prompt, stream_mode=False):
                if isinstance(event, ContentChunkEvent):
                    print(f"内容: {event.text}")
                elif isinstance(event, ToolCallCompleteEvent):
                    print(f"工具调用: {event.tool_call.name}({event.tool_call.args})")
                elif isinstance(event, AllToolResultsEvent):
                    print(f"工具结果数量: {len(event.results)}")
        else:
            async for event in agent.continue_run(prompt=prompt, stream_mode=False):
                if isinstance(event, ContentChunkEvent):
                    print(f"内容: {event.text}")
                elif isinstance(event, ToolCallCompleteEvent):
                    print(f"工具调用: {event.tool_call.name}({event.tool_call.args})")
                elif isinstance(event, AllToolResultsEvent):
                    print(f"工具结果数量: {len(event.results)}")
    
    # 返回最终的上下文消息
    return agent.context.to_openai_messages()


def normalize_message_for_comparison(msg: dict) -> dict:
    """标准化消息以便比较（移除时间戳等动态字段）"""
    normalized = msg.copy()
    
    # 移除可能导致不一致的字段
    if 'timestamp' in normalized:
        del normalized['timestamp']
    if 'message_id' in normalized:
        del normalized['message_id']
    
    # 如果有工具调用，标准化工具调用ID（它们可能不同但结构应该相同）
    if 'tool_calls' in normalized and normalized['tool_calls']:
        for i, tool_call in enumerate(normalized['tool_calls']):
            # 将tool_call_id标准化为索引
            tool_call['id'] = f"call_{i}"
    
    # 如果是工具响应消息，也标准化tool_call_id
    if normalized.get('role') == 'tool' and 'tool_call_id' in normalized:
        # 这里我们需要根据消息的顺序来标准化，暂时保持原样
        pass
    
    return normalized


def compare_contexts(stream_context: list[dict], non_stream_context: list[dict]) -> tuple[bool, str]:
    """比较两个上下文的内容是否一致"""
    
    if len(stream_context) != len(non_stream_context):
        return False, f"消息数量不一致: 流式={len(stream_context)}, 非流式={len(non_stream_context)}"
    
    for i, (stream_msg, non_stream_msg) in enumerate(zip(stream_context, non_stream_context)):
        # 标准化消息
        norm_stream = normalize_message_for_comparison(stream_msg)
        norm_non_stream = normalize_message_for_comparison(non_stream_msg)
        
        # 比较关键字段
        if norm_stream.get('role') != norm_non_stream.get('role'):
            return False, f"第{i+1}条消息角色不一致: 流式={norm_stream.get('role')}, 非流式={norm_non_stream.get('role')}"
        
        if norm_stream.get('content') != norm_non_stream.get('content'):
            return False, f"第{i+1}条消息内容不一致: 流式='{norm_stream.get('content')}', 非流式='{norm_non_stream.get('content')}'"
        
        # 比较工具调用
        stream_tools = norm_stream.get('tool_calls', [])
        non_stream_tools = norm_non_stream.get('tool_calls', [])
        
        if len(stream_tools) != len(non_stream_tools):
            return False, f"第{i+1}条消息工具调用数量不一致: 流式={len(stream_tools)}, 非流式={len(non_stream_tools)}"
        
        for j, (stream_tool, non_stream_tool) in enumerate(zip(stream_tools, non_stream_tools)):
            if stream_tool.get('type') != non_stream_tool.get('type'):
                return False, f"第{i+1}条消息第{j+1}个工具调用类型不一致"
            
            stream_func = stream_tool.get('function', {})
            non_stream_func = non_stream_tool.get('function', {})
            
            if stream_func.get('name') != non_stream_func.get('name'):
                return False, f"第{i+1}条消息第{j+1}个工具调用名称不一致"
            
            if stream_func.get('arguments') != non_stream_func.get('arguments'):
                return False, f"第{i+1}条消息第{j+1}个工具调用参数不一致"
    
    return True, "上下文完全一致"


async def test_context_consistency():
    """测试流式和非流式模式的上下文一致性"""
    print("🧪 开始测试流式和非流式模式的上下文一致性...")
    
    # 准备测试对话
    test_prompts = [
        "请告诉我北京的天气如何？",
        "现在计算一下 15 + 27 等于多少？",
        "再查询一下上海的天气，然后把上海的温度和刚才北京的温度相加。"
    ]
    
    try:
        # 测试流式模式
        print("\n=== 测试流式模式 ===")
        stream_agent = await create_test_agent()
        stream_context = await run_conversation_stream(stream_agent, test_prompts)
        
        # 测试非流式模式
        print("\n=== 测试非流式模式 ===")
        non_stream_agent = await create_test_agent()
        non_stream_context = await run_conversation_non_stream(non_stream_agent, test_prompts)
        
        # 打印上下文信息用于调试
        print(f"\n=== 上下文分析 ===")
        print(f"流式模式消息数量: {len(stream_context)}")
        print(f"非流式模式消息数量: {len(non_stream_context)}")
        
        print(f"\n流式模式上下文概览:")
        for i, msg in enumerate(stream_context):
            role = msg.get('role', '未知')
            content_preview = (msg.get('content', '') or '')[:50] + "..." if len((msg.get('content', '') or '')) > 50 else (msg.get('content', '') or '')
            tool_calls_count = len(msg.get('tool_calls', []))
            print(f"  {i+1}. {role}: '{content_preview}' (工具调用: {tool_calls_count})")
        
        print(f"\n非流式模式上下文概览:")
        for i, msg in enumerate(non_stream_context):
            role = msg.get('role', '未知')
            content_preview = (msg.get('content', '') or '')[:50] + "..." if len((msg.get('content', '') or '')) > 50 else (msg.get('content', '') or '')
            tool_calls_count = len(msg.get('tool_calls', []))
            print(f"  {i+1}. {role}: '{content_preview}' (工具调用: {tool_calls_count})")
        
        # 比较上下文
        is_consistent, message = compare_contexts(stream_context, non_stream_context)
        
        print(f"\n=== 测试结果 ===")
        if is_consistent:
            print("✅ 测试通过: 流式和非流式模式的上下文完全一致!")
            print(f"详细信息: {message}")
            return True
        else:
            print("❌ 测试失败: 流式和非流式模式的上下文不一致!")
            print(f"差异: {message}")
            
            # 输出详细的上下文内容用于调试
            print(f"\n=== 详细对比 ===")
            print("流式模式完整上下文:")
            for i, msg in enumerate(stream_context):
                print(f"  [{i+1}] {msg}")
            
            print("\n非流式模式完整上下文:")
            for i, msg in enumerate(non_stream_context):
                print(f"  [{i+1}] {msg}")
            
            return False
            
    except Exception as e:
        print(f"❌ 测试执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_basic_context_operations():
    """测试基础的上下文操作"""
    print("\n🧪 测试基础上下文操作...")
    
    try:
        agent = await create_test_agent()
        
        # 检查初始状态
        initial_messages = len(agent.context.messages)
        print(f"初始消息数量: {initial_messages}")
        
        # 添加一条用户消息
        agent.context.add_user_message("测试消息")
        after_user = len(agent.context.messages)
        print(f"添加用户消息后: {after_user}")
        
        # 转换为OpenAI格式
        openai_messages = agent.context.to_openai_messages()
        print(f"OpenAI格式消息数量: {len(openai_messages)}")
        
        # 验证消息内容
        user_msg = next((msg for msg in openai_messages if msg.get('role') == 'user'), None)
        if user_msg and user_msg.get('content') == '测试消息':
            print("✅ 基础上下文操作测试通过")
            return True
        else:
            print("❌ 基础上下文操作测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 基础上下文操作测试出错: {e}")
        return False


def main():
    """主测试函数"""
    print("🤖 开始上下文一致性测试...")
    
    async def run_all_tests():
        results = []
        
        # 基础上下文操作测试
        basic_result = await test_basic_context_operations()
        results.append(("基础上下文操作", basic_result))
        
        # 上下文一致性测试
        consistency_result = await test_context_consistency()
        results.append(("上下文一致性", consistency_result))
        
        # 统计结果
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        print(f"\n📊 测试结果汇总:")
        for test_name, result in results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"  {test_name}: {status}")
        
        print(f"\n总体结果: {passed}/{total} 通过")
        
        if passed == total:
            print("🎉 所有测试都通过了!")
        else:
            print("⚠️  部分测试失败，请检查上下文管理逻辑")
            
        return passed == total
    
    return asyncio.run(run_all_tests())


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 