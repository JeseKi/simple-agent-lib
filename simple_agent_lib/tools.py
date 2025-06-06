"""
工具装饰器模块

该模块包含:
- @tool装饰器实现
- 工具注册表管理
- 动态Schema生成

公开接口:
- tool: 工具装饰器
- get_tool_schemas: 获取所有工具Schema
- get_tool_registry: 获取工具注册表

内部方法:
- _generate_parameter_schema: 生成参数Schema
- _get_type_schema: 获取类型Schema
"""

import inspect
from typing import (
    Dict,
    Any,
    Callable,
    List,
    get_type_hints,
    get_origin,
    get_args,
    Union,
)
from pydantic import BaseModel

# 工具注册表
_TOOL_REGISTRY: Dict[str, Callable] = {}
_TOOL_SCHEMAS: List[Dict[str, Any]] = []


def _get_type_schema(param_type: type) -> Dict[str, Any]:
    """
    根据Python类型生成JSON Schema

    Args:
        param_type: Python类型

    Returns:
        Dict[str, Any]: JSON Schema定义
    """
    # 处理Union类型（包括Optional）
    origin = get_origin(param_type)
    if origin is Union:
        args = get_args(param_type)
        # 处理Optional类型 (Union[T, None])
        if len(args) == 2 and type(None) in args:
            non_none_type = args[0] if args[1] is type(None) else args[1]
            schema = _get_type_schema(non_none_type)
            # Optional类型在required中处理，这里不需要特殊标记
            return schema
        else:
            # 其他Union类型，使用anyOf
            return {"anyOf": [_get_type_schema(arg) for arg in args]}

    # 基础类型映射
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        List: {"type": "array"},
        Dict: {"type": "object"},
    }

    # 检查是否为Pydantic模型
    if inspect.isclass(param_type) and issubclass(param_type, BaseModel):
        # 如果是Pydantic模型，使用其schema
        try:
            return param_type.model_json_schema()
        except Exception as e:
            print(
                f"Error generating schema for Pydantic model {param_type.__name__}: {e}"
            )
            return {
                "type": "object",
                "description": f"Pydantic模型: {param_type.__name__}",
            }

    # 处理泛型类型
    if origin:
        if origin is list or origin is List:
            args = get_args(param_type)
            if args:
                return {"type": "array", "items": _get_type_schema(args[0])}
            return {"type": "array"}
        elif origin is dict or origin is Dict:
            return {"type": "object"}

    # 查找基础类型
    return type_mapping.get(
        param_type, {"type": "string", "description": f"未知类型: {param_type}"}
    )


def _generate_parameter_schema(func: Callable) -> Dict[str, Any]:
    """
    根据函数签名生成参数Schema

    Args:
        func: 要分析的函数

    Returns:
        Dict[str, Any]: 参数Schema
    """
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    properties = {}
    required = []

    for param_name, param in signature.parameters.items():
        # 跳过self和cls参数
        if param_name in ("self", "cls"):
            continue

        param_type = type_hints.get(param_name, str)
        param_schema = _get_type_schema(param_type)

        # 添加参数描述（如果有的话）
        if func.__doc__:
            # 简单的docstring解析，寻找Args部分
            doc_lines = func.__doc__.split("\n")
            in_args_section = False
            for line in doc_lines:
                line = line.strip()
                if line.startswith("Args:"):
                    in_args_section = True
                    continue
                elif line.startswith("Returns:") or line.startswith("Raises:"):
                    in_args_section = False
                    continue

                if in_args_section and f"{param_name}" in line:
                    # 提取描述部分
                    if ":" in line:
                        desc_part = line.split(":", 2)[-1].strip()
                        if desc_part:
                            param_schema["description"] = desc_part
                    break

        properties[param_name] = param_schema

        # 检查是否为必需参数
        if param.default is inspect.Parameter.empty:
            # 检查是否为Optional类型
            origin = get_origin(param_type)
            if origin is Union:
                args = get_args(param_type)
                if not (len(args) == 2 and type(None) in args):
                    # 不是Optional类型，是必需的
                    required.append(param_name)
            else:
                # 不是Union类型，是必需的
                required.append(param_name)

    return {"type": "object", "properties": properties, "required": required}


def tool(func: Callable) -> Callable:
    """
    将函数注册为可调用工具的装饰器

    Args:
        func: 要注册的函数

    Returns:
        Callable: 原函数（未修改）
    """
    func_name = func.__name__

    # 生成参数Schema
    parameters_schema = _generate_parameter_schema(func)

    # 提取函数描述
    description = func.__doc__ or f"工具函数 {func_name}"
    # 清理描述，只取第一行或Args之前的部分
    if description:
        lines = description.split("\n")
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith("Args:"):
                break
            if line:
                clean_lines.append(line)
        description = (
            " ".join(clean_lines) if clean_lines else description.split("\n")[0].strip()
        )

    # 构建工具Schema
    tool_schema = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": description,
            "parameters": parameters_schema,
        },
    }

    # 注册工具
    _TOOL_REGISTRY[func_name] = func
    _TOOL_SCHEMAS.append(tool_schema)

    # 将schema信息存储在函数的属性中（使用setattr避免类型检查问题）
    setattr(func, "_tool_schema", tool_schema)
    setattr(func, "_tool_name", func_name)

    # 使用日志记录工具注册
    from .logger_config import log_tool_registration, log_schema_generation

    log_tool_registration(func_name, list(parameters_schema["properties"].keys()))
    log_schema_generation(func_name, tool_schema)

    return func


def get_tool_schemas() -> List[Dict[str, Any]]:
    """
    获取所有注册工具的Schema

    Returns:
        List[Dict[str, Any]]: 工具Schema列表
    """
    return _TOOL_SCHEMAS.copy()


def get_tool_registry() -> Dict[str, Callable]:
    """
    获取工具注册表

    Returns:
        Dict[str, Callable]: 工具名称到函数的映射
    """
    return _TOOL_REGISTRY.copy()


def clear_tools():
    """清空所有注册的工具"""
    global _TOOL_REGISTRY, _TOOL_SCHEMAS
    _TOOL_REGISTRY.clear()
    _TOOL_SCHEMAS.clear()
