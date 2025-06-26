"""
日志配置模块

该模块包含:
- 统一的日志配置（不影响主应用程序）
- 不同级别的日志记录器
- 文件和控制台输出配置
- 库专用日志启用/禁用控制

公开接口:
- enable_logging: 启用或禁用库的日志输出
- disable_logging: 完全禁用库的日志输出
- is_logging_enabled: 检查日志是否启用
- setup_logger: 设置日志器
- get_logger: 获取日志器实例
- log_tool_registration: 记录工具注册
- log_agent_iteration: 记录Agent迭代
- log_tool_execution: 记录工具执行
- log_llm_interaction: 记录LLM交互
- log_agent_event: 记录Agent事件
- log_tool_call_parsing: 记录工具调用解析
- log_schema_generation: 记录Schema生成
- log_http_request: 记录HTTP请求
- log_agent_completion: 记录Agent完成

内部方法:
- _LOGGING_ENABLED: 全局日志启用状态
- _CURRENT_LOG_LEVEL: 当前日志级别
- _HANDLER_IDS: 库专用的处理器ID列表
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from loguru import logger

# 库专用的日志控制状态
_LOGGING_ENABLED = False  # 默认禁用日志
_CURRENT_LOG_LEVEL = "INFO"
# 存储库专用的处理器ID，用于精确管理自己的处理器
_HANDLER_IDS: List[int] = []
# 库的命名空间
_LIBRARY_NAMESPACE = "simple_agent_lib"


def _library_log_filter(record):
    """
    库专用的日志过滤器，只允许库命名空间的日志通过
    """
    extra_name = record.get("extra", {}).get("name", "")
    if extra_name is None:
        extra_name = ""
    return str(extra_name).startswith(_LIBRARY_NAMESPACE)


def _clear_library_handlers():
    """
    清理库专用的日志处理器，但不改变日志启用状态
    """
    global _HANDLER_IDS
    
    # 只移除库自己添加的处理器
    for handler_id in _HANDLER_IDS:
        try:
            logger.remove(handler_id)
        except ValueError:
            # 处理器可能已经被移除，忽略错误
            pass
    
    _HANDLER_IDS.clear()


def enable_logging(enabled: bool = True, log_level: str = "INFO") -> None:
    """
    启用或禁用库的日志输出
    
    注意：此函数只管理simple_agent_lib库自己的日志处理器，
    不会影响主应用程序的日志配置。

    Args:
        enabled: 是否启用日志
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global _LOGGING_ENABLED, _CURRENT_LOG_LEVEL
    _LOGGING_ENABLED = enabled
    _CURRENT_LOG_LEVEL = log_level

    if enabled:
        setup_logger(log_level=log_level)
    else:
        disable_logging()


def disable_logging() -> None:
    """
    完全禁用库的日志输出
    
    注意：此函数只移除simple_agent_lib库自己添加的日志处理器，
    不会影响主应用程序的日志配置。
    """
    global _LOGGING_ENABLED
    _LOGGING_ENABLED = False
    
    # 清理库专用的处理器
    _clear_library_handlers()


def is_logging_enabled() -> bool:
    """检查日志是否启用"""
    return _LOGGING_ENABLED


def setup_logger(
    log_file: str = "simple_agent_lib.log",
    log_level: str = "INFO",
    console_output: bool = True,
    file_output: bool = False,  # 默认不输出到文件，减少对主应用的影响
    max_file_size: str = "10 MB",
    rotation_count: int = 3,
) -> None:
    """
    设置日志配置
    
    注意：此函数只添加simple_agent_lib库专用的日志处理器，
    不会影响主应用程序的日志配置。

    Args:
        log_file: 日志文件路径
        log_level: 日志级别
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        max_file_size: 文件最大大小
        rotation_count: 文件轮转数量
    """
    global _CURRENT_LOG_LEVEL, _LOGGING_ENABLED
    _CURRENT_LOG_LEVEL = log_level

    # 先清理现有的库专用处理器（但不改变_LOGGING_ENABLED状态）
    _clear_library_handlers()

    # 如果日志被禁用，直接返回
    if not _LOGGING_ENABLED:
        return

    # 控制台日志配置
    if console_output:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            f"<cyan>[{_LIBRARY_NAMESPACE}]</cyan> | "
            "<cyan>{extra[name]}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        handler_id = logger.add(
            sys.stdout,
            format=console_format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
            filter=_library_log_filter
        )
        _HANDLER_IDS.append(handler_id)

    # 文件日志配置
    if file_output:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            f"[{_LIBRARY_NAMESPACE}] | "
            "{extra[name]}:{function}:{line} | "
            "{message}"
        )
        handler_id = logger.add(
            Path(Path.cwd() / f"{_LIBRARY_NAMESPACE}_{log_file}"),
            format=file_format,
            level=log_level,
            rotation=max_file_size,
            retention=rotation_count,
            compression="zip",
            encoding="utf-8",
            backtrace=True,
            diagnose=True,
            filter=_library_log_filter
        )
        _HANDLER_IDS.append(handler_id)


def get_logger(name: Optional[str] = None):
    """
    获取日志器实例
    
    注意：返回的logger使用库专用的命名空间，不会与主应用程序的日志混淆。

    Args:
        name: 日志器名称（将自动添加库命名空间前缀）

    Returns:
        logger实例
    """
    if not _LOGGING_ENABLED:
        # 返回一个空的日志器，所有方法都不执行任何操作
        class NoOpLogger:
            def debug(self, *args, **kwargs):
                pass

            def info(self, *args, **kwargs):
                pass

            def warning(self, *args, **kwargs):
                pass

            def error(self, *args, **kwargs):
                pass

            def critical(self, *args, **kwargs):
                pass

            def success(self, *args, **kwargs):
                pass

            def trace(self, *args, **kwargs):
                pass

            def bind(self, **kwargs):
                return self

        return NoOpLogger()

    # 使用库专用的命名空间
    if name:
        logger_name = f"{_LIBRARY_NAMESPACE}.{name}"
    else:
        logger_name = _LIBRARY_NAMESPACE
    
    return logger.bind(name=logger_name)


# 预定义的日志记录函数


def log_tool_registration(
    tool_name: str, params: list, schema_info: Optional[Dict] = None
):
    """记录工具注册信息"""
    if not _LOGGING_ENABLED:
        return
    lib_logger = get_logger("工具系统")
    lib_logger.info(f"工具 '{tool_name}' 已注册，参数: {params}")
    if schema_info:
        lib_logger.debug(f"工具 '{tool_name}' Schema详情: {schema_info}")


def log_agent_iteration(iteration: int, total_iterations: Optional[int] = None):
    """记录Agent迭代信息"""
    if not _LOGGING_ENABLED:
        return
    lib_logger = get_logger("智能体")
    if total_iterations and total_iterations > 0:
        lib_logger.info(f"--- Agent迭代 {iteration}/{total_iterations} ---")
    else:
        lib_logger.info(f"--- Agent迭代 {iteration} ---")


def log_tool_execution(
    tool_name: str,
    args: Dict[str, Any],
    success: bool,
    result: Any = None,
    error: Optional[str] = None,
):
    """记录工具执行信息"""
    if not _LOGGING_ENABLED:
        return
    lib_logger = get_logger("工具执行")
    if success:
        lib_logger.success(f"工具 '{tool_name}' 执行成功，参数: {args}")
        if result is not None:
            # 限制结果长度避免日志过大
            result_str = str(result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "..."
            lib_logger.debug(f"工具 '{tool_name}' 结果: {result_str}")
    else:
        lib_logger.error(f"工具 '{tool_name}' 执行失败，参数: {args}，错误: {error}")


def log_llm_interaction(
    action: str, details: Optional[str] = None, error: Optional[str] = None
):
    """记录LLM交互信息"""
    if not _LOGGING_ENABLED:
        return
    lib_logger = get_logger("LLM交互")
    if error:
        lib_logger.error(f"LLM {action} 失败: {error}")
        if details:
            lib_logger.debug(f"详细信息: {details}")
    else:
        lib_logger.info(f"LLM {action}")
        if details:
            lib_logger.debug(f"详细信息: {details}")


def log_agent_event(
    event_type: str, content: str, llm_response_id: Optional[str] = None
):
    """记录Agent事件"""
    if not _LOGGING_ENABLED:
        return
    lib_logger = get_logger("Agent事件")
    if llm_response_id:
        lib_logger.debug(f"[{event_type}] ({llm_response_id}) {content}")
    else:
        lib_logger.debug(f"[{event_type}] {content}")


def log_tool_call_parsing(
    success: bool, tool_name: str, args: Dict[str, Any], error: Optional[str] = None
):
    """记录工具调用解析"""
    if not _LOGGING_ENABLED:
        return
    lib_logger = get_logger("工具解析")
    if success:
        lib_logger.debug(f"工具调用解析成功: {tool_name}({args})")
    else:
        lib_logger.warning(f"工具调用解析失败: {tool_name}，错误: {error}")


def log_schema_generation(tool_name: str, schema: Dict[str, Any]):
    """记录Schema生成"""
    if not _LOGGING_ENABLED:
        return
    lib_logger = get_logger("Schema生成")
    lib_logger.debug(f"为工具 '{tool_name}' 生成Schema")
    lib_logger.trace(f"Schema内容: {schema}")


def log_http_request(
    method: str,
    url: str,
    status_code: Optional[int] = None,
    error: Optional[str] = None,
):
    """记录HTTP请求"""
    if not _LOGGING_ENABLED:
        return
    lib_logger = get_logger("HTTP请求")
    if error:
        lib_logger.error(f"{method} {url} 失败: {error}")
    else:
        lib_logger.info(f"{method} {url} 响应码: {status_code}")


def log_agent_completion(
    reason: str, iterations_used: int, max_iterations: Optional[int] = None
):
    """记录Agent完成信息"""
    if not _LOGGING_ENABLED:
        return
    lib_logger = get_logger("智能体")
    if max_iterations and max_iterations > 0:
        lib_logger.info(
            f"Agent执行完成，原因: {reason}，使用迭代: {iterations_used}/{max_iterations}"
        )
    else:
        lib_logger.info(f"Agent执行完成，原因: {reason}，使用迭代: {iterations_used}")


# 库初始化时不自动设置日志器，等待用户显式启用
# 这样可以避免在导入时就影响主应用程序的日志配置
