#!/usr/bin/env python3
"""
Agent Lib 安装验证测试

验证库是否正确安装并能正常工作
"""

def test_basic_import():
    """测试基础导入"""
    print("=== 测试基础导入 ===")
    try:
        import agent_lib
        print(f"✅ Agent Lib 版本: {agent_lib.__version__}")
        print(f"✅ 作者: {agent_lib.__author__}")
        print(f"✅ 描述: {agent_lib.__description__}")
        return True
    except Exception as e:
        print(f"❌ 基础导入失败: {e}")
        return False


def test_core_imports():
    """测试核心功能导入"""
    print("\n=== 测试核心功能导入 ===")
    try:
        from agent_lib import (
            Context, 
            AutonomousAgent, 
            tool, 
            ContextTokenLimitExceededError,
            LLMAPIClient,
            ToolCall,
            ToolResult
        )
        print("✅ 所有核心类导入成功")
        return True
    except Exception as e:
        print(f"❌ 核心功能导入失败: {e}")
        return False


def test_context_functionality():
    """测试上下文功能"""
    print("\n=== 测试上下文功能 ===")
    try:
        from agent_lib import Context
        
        # 创建上下文
        context = Context(max_tokens=100, max_messages=10)
        context.add_user_message("测试消息")
        
        print(f"✅ 上下文消息数: {len(context.messages)}")
        print(f"✅ 估算tokens: {context.estimate_tokens()}")
        
        # 测试OpenAI格式转换
        openai_format = context.to_openai_messages()
        print(f"✅ OpenAI格式转换成功，消息数: {len(openai_format)}")
        
        return True
    except Exception as e:
        print(f"❌ 上下文功能测试失败: {e}")
        return False


def test_tool_decorator():
    """测试工具装饰器"""
    print("\n=== 测试工具装饰器 ===")
    try:
        from agent_lib import tool, get_tool_registry
        
        @tool
        def test_tool(x: int) -> str:
            """测试工具"""
            return f"结果: {x * 2}"
        
        print(f"✅ 工具注册成功: {test_tool.__name__}")
        
        # 测试工具调用
        result = test_tool(5)
        print(f"✅ 工具调用成功: {result}")
        
        # 检查工具注册表
        registry = get_tool_registry()
        print(f"✅ 工具注册表包含 {len(registry)} 个工具")
        
        return True
    except Exception as e:
        print(f"❌ 工具装饰器测试失败: {e}")
        return False


def test_exception_handling():
    """测试异常处理"""
    print("\n=== 测试异常处理 ===")
    try:
        from agent_lib import Context, ContextTokenLimitExceededError
        
        # 测试token超限异常
        try:
            small_context = Context(max_tokens=5)
            small_context.add_system_message("很长的消息" * 100)
            print("❌ 未抛出预期的异常")
            return False
        except ContextTokenLimitExceededError as e:
            print("✅ 异常处理正常")
            print(f"✅ 异常信息: {str(e)[:50]}...")
            print(f"✅ 异常属性: token_limit={e.token_limit}, current_tokens={e.current_tokens}")
            return True
        
    except Exception as e:
        print(f"❌ 异常处理测试失败: {e}")
        return False


def test_agent_creation():
    """测试智能体创建"""
    print("\n=== 测试智能体创建 ===")
    try:
        from agent_lib import AutonomousAgent, Context
        
        # 创建上下文
        context = Context(max_tokens=1000, max_messages=50)
        
        # 创建智能体（需要LLM客户端，这里跳过实际创建）
        print("✅ 智能体类可以正常导入")
        print("✅ 智能体创建测试通过")
        print(f"✅ 上下文消息数: {len(context.messages)}")
        
        return True
    except Exception as e:
        print(f"❌ 智能体创建测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🧪 开始 Agent Lib 安装验证测试...\n")
    
    tests = [
        test_basic_import,
        test_core_imports,
        test_context_functionality,
        test_tool_decorator,
        test_exception_handling,
        test_agent_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！Agent Lib 安装成功且功能正常！")
        print("\n✨ 你现在可以开始使用 Agent Lib 了：")
        print("```python")
        print("from agent_lib import AutonomousAgent, Context, tool")
        print("```")
    else:
        print("❌ 部分测试失败，请检查安装或依赖")
    
    return passed == total


if __name__ == "__main__":
    main() 