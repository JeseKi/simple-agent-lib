"""
智能体框架使用示例

该文件展示如何使用自主智能体框架，参考agno框架的设计模式

示例包含:
- 天气查询工具
- Python代码执行工具（简化版）
- HTTP请求工具
- 完整的Agent使用示例

基于agno_test.py的实际使用场景
"""

import asyncio
import os
import random
import http.client
import subprocess
import tempfile
from typing import Dict, Any
from pydantic import BaseModel
from loguru import logger

# 导入我们的框架组件
from simple_agent_lib.schemas import (
    ReasoningChunkEvent,
    ContentChunkEvent,
    ToolCallCompleteEvent,
    AllToolResultsEvent,
)
from simple_agent_lib.tools import tool, get_tool_schemas
from simple_agent_lib.core import LLMAPIClient, AutonomousAgent

from dotenv import load_dotenv

load_dotenv(".env")

# === 工具定义 ===


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
    print(f"[工具执行] 获取 {city} 的天气信息")
    # raise Exception("无法获取天气信息，请检查您的网络连接！") # 测试错误处理
    weather_conditions = ["晴天", "多云", "阴天", "小雨", "大雨", "暴风雨"]
    return WeatherResponse(
        city=city,
        temperature=random.randint(10, 30),
        weather=random.choice(weather_conditions),
    )


@tool
def run_python_code(code: str) -> Dict[str, Any]:
    """执行Python代码并返回结果

    Args:
        code: 要执行的Python代码

    Returns:
        Dict[str, Any]: 执行结果，包含输出和可能的错误
    """
    print(f"[工具执行] 运行Python代码:\n{code}")

    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        # 执行代码
        result = subprocess.run(
            ["python", temp_file], capture_output=True, text=True, timeout=10
        )

        # 清理临时文件
        os.unlink(temp_file)

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "代码执行超时",
            "returncode": -1,
            "success": False,
        }
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1, "success": False}


@tool
def http_get_request(hostname: str, path: str = "/", port: int = 80) -> Dict[str, Any]:
    """发送HTTP GET请求

    Args:
        hostname: 主机名
        path: 请求路径
        port: 端口号

    Returns:
        Dict[str, Any]: HTTP响应信息
    """
    print(f"[工具执行] 发送HTTP请求到 {hostname}:{port}{path}")

    try:
        # 使用http.client发送请求
        if port == 443:
            conn = http.client.HTTPSConnection(hostname)
        else:
            conn = http.client.HTTPConnection(hostname, port)

        conn.request("GET", path)
        response = conn.getresponse()

        # 读取响应内容（限制大小避免过大的响应）
        content = response.read(10000).decode("utf-8", errors="ignore")

        conn.close()

        return {
            "status_code": response.status,
            "reason": response.reason,
            "headers": dict(response.getheaders()),
            "content": content[:1000] + "..." if len(content) > 1000 else content,
            "content_length": len(content),
        }
    except Exception as e:
        return {"error": str(e), "status_code": 0, "reason": "请求失败"}


@tool
def calculate_power(base: float, exponent: float) -> Dict[str, Any]:
    """计算幂运算

    Args:
        base: 底数
        exponent: 指数

    Returns:
        Dict[str, Any]: 计算结果
    """
    print(f"[工具执行] 计算 {base} 的 {exponent} 次幂")

    try:
        result = base**exponent
        return {
            "base": base,
            "exponent": exponent,
            "result": result,
            "calculation": f"{base}^{exponent} = {result}",
        }
    except Exception as e:
        return {"error": str(e), "base": base, "exponent": exponent, "result": None}


# === Agent设置和运行 ===


async def create_agent() -> AutonomousAgent:
    """创建并配置智能体"""

    # 配置LLM API客户端
    # 注意：您需要设置实际的API信息
    OPENAI_API_BASE = os.getenv(
        "OPENAI_API_BASE", "https://api.openai.com/v1"
    )  # 例如: "http://localhost:8000/v1"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-1234567890")
    LLM_MODEL_NAME = "Qwen3-32B"  # 或您使用的模型名称

    if (
        OPENAI_API_BASE == "YOUR_LLM_API_BASE_URL"
        or OPENAI_API_KEY == "YOUR_LLM_API_KEY"
    ):
        print("⚠️  请在create_agent函数中配置您的LLM API信息")
        print("   设置 LLM_BASE_URL 和 LLM_API_KEY")
        # 为了演示，我们使用模拟的配置
        OPENAI_API_BASE = "http://localhost:8000/v1"
        OPENAI_API_KEY = "demo-key"

    # 创建LLM客户端
    llm_client = LLMAPIClient(
        base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY, model_name=LLM_MODEL_NAME
    )

    # 创建Agent，传入所有注册的工具
    agent = AutonomousAgent(
        llm_api_client=llm_client,
        tools=[get_weather, run_python_code, http_get_request, calculate_power],
    )

    return agent


async def run_example_task():
    """运行示例任务（基于agno_test.py中的任务）"""

    agent = await create_agent()

    # 任务提示（来自agno_test.py）
    task_prompt = """
    使用Python来调用 http.client 来查看一下微软域名的内容看看，
    然后告诉我微软总部的天气和北京的天气怎么样，
    然后计算微软总部天气温度数值的北京天气温度数值次幂。
    
    请一步一步的处理这个请求，每次调用工具前都告诉我你要做什么，
    并且每次得到工具的结果后告诉我目前的进度，再进行下一步。
    """

    print("🚀 开始执行任务...")
    print(f"📝 任务: {task_prompt.strip()}")
    print("\n" + "=" * 50)

    # 运行Agent
    try:
        has_reasoning = False
        reasoning_content = []

        async for event in agent.run(initial_prompt=task_prompt, stream_mode=True):
            if isinstance(event, ReasoningChunkEvent):
                if not has_reasoning:
                    print("\n🤔 [推理过程]:", end="")
                    has_reasoning = True
                print(event.text, end="", flush=True)
                reasoning_content.append(event.text)
            elif isinstance(event, ContentChunkEvent):
                if has_reasoning:
                    print("\n\n💬 [AI回复]:", end="")
                    has_reasoning = False
                print(event.text, end="", flush=True)
            elif isinstance(event, ToolCallCompleteEvent):
                print(f"\n🔧 调用工具: {event.tool_call.name}")
            elif isinstance(event, AllToolResultsEvent):
                print("✅ 工具执行完成")
                
        logger.info(f"完整上下文：{agent.context.messages}")

    except Exception as e:
        print(f"❌ 任务执行出错: {e}")
        import traceback

        traceback.print_exc()


async def simple_weather_example():
    """简单的天气查询示例"""

    agent = await create_agent()

    print("\n🌤️  简单天气查询示例")
    print("=" * 30)

    prompt = "请告诉我北京和上海的天气情况，并比较一下哪个城市更适合外出。"

    # 用于跟踪是否有推理内容
    has_reasoning = False
    reasoning_content = []

    async for event in agent.run(initial_prompt=prompt, stream_mode=True):
        if isinstance(event, ReasoningChunkEvent):
            if not has_reasoning:
                print("\n🤔 [推理过程]:", end="")
                has_reasoning = True
            print(event.text, end="", flush=True)
            reasoning_content.append(event.text)
        elif isinstance(event, ContentChunkEvent):
            if has_reasoning:
                print("\n\n💬 [AI回复]:", end="")
                has_reasoning = False
            print(event.text, end="", flush=True)
        elif isinstance(event, ToolCallCompleteEvent):
            print(f"\n🔧 调用工具: {event.tool_call.name}")
        elif isinstance(event, AllToolResultsEvent):
            print("✅ 工具执行完成")
            
    logger.info(f"完整上下文：{agent.context.messages}")

    # 如果捕获到了推理内容，显示总结
    if reasoning_content:
        print(f"\n\n📝 本次会话共捕获推理内容: {len(reasoning_content)} 个片段")
        full_reasoning = "".join(reasoning_content)
        print(f"📊 推理内容总长度: {len(full_reasoning)} 字符")
    else:
        print("\n\n📝 本次会话未检测到推理内容（可能使用的是非推理模型）")


def show_registered_tools():
    """显示所有注册的工具信息"""
    print("\n🛠️  已注册的工具:")
    print("=" * 20)

    # registry = get_tool_registry()
    schemas = get_tool_schemas()

    for schema in schemas:
        func_info = schema["function"]
        name = func_info["name"]
        description = func_info["description"]
        params = func_info["parameters"]["properties"]

        print(f"📌 {name}")
        print(f"   描述: {description}")
        print(f"   参数: {list(params.keys())}")
        print()


async def main():
    """主函数"""
    print("🤖 自主智能体框架示例")
    print("基于agno框架设计模式")
    print("=" * 40)

    # 显示工具信息
    show_registered_tools()

    # 运行简单示例
    await simple_weather_example()

    print("\n" + "=" * 50)
    print("主要任务演示:")

    await run_example_task()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())
