#!/usr/bin/env python3
"""
测试工具调用时文本内容的保存
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
    ContentChunkEvent,
    ToolCallCompleteEvent,
    AllToolResultsEvent,
)


@tool
def simple_calculator(a: float, b: float) -> Dict[str, Any]:
    """简单计算器"""
    return {"result": a + b, "operation": f"{a} + {b} = {a + b}"}


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
        tools=[simple_calculator],
        system_prompt="你是一个数学助手，在使用工具前后都要说明你在做什么。",
        debug_mode=True
    )
    
    return agent


async def test_tool_call_with_text():
    """测试工具调用时的文本内容保存"""
    print("🧪 测试工具调用时的文本内容保存...")
    
    agent = await create_test_agent()
    
    # 使用一个会触发工具调用并产生文本说明的提示
    prompt = "请计算 5 + 3，在计算前说明你要做什么，在计算后说明结果。"
    print(f"用户输入: {prompt}")
    
    content_chunks = []
    tool_calls = []
    
    print("\n=== 流式模式 ===")
    async for event in agent.run(initial_prompt=prompt, stream_mode=True):
        if isinstance(event, ContentChunkEvent):
            content_chunks.append(event.text)
            print(f"[内容] {event.text}", end="")
        elif isinstance(event, ToolCallCompleteEvent):
            tool_calls.append(event.tool_call)
            print(f"\n[工具] {event.tool_call.name}({event.tool_call.args})")
        elif isinstance(event, AllToolResultsEvent):
            print(f"\n[结果] 工具执行完成")
    
    print(f"\n\n收集到的内容: {''.join(content_chunks)}")
    print(f"工具调用次数: {len(tool_calls)}")
    
    # 分析上下文
    context_messages = agent.context.to_openai_messages()
    print(f"\n=== 上下文分析 ===")
    print(f"消息总数: {len(context_messages)}")
    
    for i, msg in enumerate(context_messages, 1):
        role = msg.get('role', '未知')
        content = (msg.get('content', '') or '')[:100] + "..." if len((msg.get('content', '') or '')) > 100 else (msg.get('content', '') or '')
        tool_calls_count = len(msg.get('tool_calls', []))
        
        print(f"  {i}. {role}: '{content}' (工具调用: {tool_calls_count})")
    
    # 检查关键问题：是否有assistant消息同时包含文本内容和工具调用？
    assistant_messages = [msg for msg in context_messages if msg.get('role') == 'assistant']
    
    print(f"\n=== 关键检查 ===")
    for i, msg in enumerate(assistant_messages, 1):
        has_content = bool(msg.get('content'))
        has_tools = bool(msg.get('tool_calls'))
        
        if has_tools:
            if has_content:
                print(f"✅ 助手消息 {i}: 同时包含工具调用和文本内容")
                print(f"   文本: '{msg.get('content', '')[:50]}...'")
            else:
                print(f"❌ 助手消息 {i}: 只有工具调用，缺少文本内容")
        elif has_content:
            print(f"✅ 助手消息 {i}: 纯文本回复")
    
    # 检查是否存在问题
    tool_with_text_count = sum(1 for msg in assistant_messages 
                               if msg.get('tool_calls') and msg.get('content'))
    tool_without_text_count = sum(1 for msg in assistant_messages 
                                  if msg.get('tool_calls') and not msg.get('content'))
    
    print(f"\n📊 统计:")
    print(f"有工具调用且有文本的助手消息: {tool_with_text_count}")
    print(f"有工具调用但无文本的助手消息: {tool_without_text_count}")
    
    if tool_without_text_count > 0 and len(content_chunks) > 0:
        print(f"⚠️  发现问题：LLM产生了文本内容（{len(content_chunks)}个块），但部分工具调用消息中没有包含文本内容！")
        return False
    else:
        print(f"✅ 测试通过：工具调用和文本内容都被正确保存")
        return True


def main():
    """主函数"""
    return asyncio.run(test_tool_call_with_text())


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 