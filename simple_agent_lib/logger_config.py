"""
日志配置模块

该模块包含:
- 统一的日志配置
- 不同级别的日志记录器
- 文件和控制台输出配置

公开接口:
- setup_logger: 设置日志器
- get_logger: 获取日志器实例
- log_tool_registration: 记录工具注册
- log_agent_iteration: 记录Agent迭代
- log_tool_execution: 记录工具执行
- log_llm_interaction: 记录LLM交互

内部方法:
- _configure_file_logging: 配置文件日志
- _configure_console_logging: 配置控制台日志
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

# 移除默认的logger配置
logger.remove()

def setup_logger(
    log_file: str = "temp.log", 
    log_level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
    max_file_size: str = "10 MB",
    rotation_count: int = 3
) -> None:
    """
    设置日志配置
    
    Args:
        log_file: 日志文件路径
        log_level: 日志级别
        console_output: 是否输出到控制台
        file_output: 是否输出到文件
        max_file_size: 文件最大大小
        rotation_count: 文件轮转数量
    """
    
    # 控制台日志配置
    if console_output:
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stdout,
            format=console_format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
    
    # 文件日志配置
    if file_output:
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        logger.add(
            Path(Path.cwd() / log_file),
            format=file_format,
            level=log_level,
            rotation=max_file_size,
            retention=rotation_count,
            compression="zip",
            encoding="utf-8",
            backtrace=True,
            diagnose=True
        )

def get_logger(name: Optional[str] = None):
    """
    获取日志器实例
    
    Args:
        name: 日志器名称
        
    Returns:
        logger实例
    """
    if name:
        return logger.bind(name=name)
    return logger

# 预定义的日志记录函数

def log_tool_registration(tool_name: str, params: list, schema_info: Optional[Dict] = None):
    """记录工具注册信息"""
    logger = get_logger("工具系统")
    logger.info(f"工具 '{tool_name}' 已注册，参数: {params}")
    if schema_info:
        logger.debug(f"工具 '{tool_name}' Schema详情: {schema_info}")

def log_agent_iteration(iteration: int, total_iterations: Optional[int] = None):
    """记录Agent迭代信息"""
    logger = get_logger("智能体")
    if total_iterations and total_iterations > 0:
        logger.info(f"--- Agent迭代 {iteration}/{total_iterations} ---")
    else:
        logger.info(f"--- Agent迭代 {iteration} ---")

def log_tool_execution(tool_name: str, args: Dict[str, Any], success: bool, result: Any = None, error: Optional[str] = None):
    """记录工具执行信息"""
    logger = get_logger("工具执行")
    if success:
        logger.success(f"工具 '{tool_name}' 执行成功，参数: {args}")
        if result is not None:
            # 限制结果长度避免日志过大
            result_str = str(result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "..."
            logger.debug(f"工具 '{tool_name}' 结果: {result_str}")
    else:
        logger.error(f"工具 '{tool_name}' 执行失败，参数: {args}，错误: {error}")

def log_llm_interaction(action: str, details: Optional[str] = None, error: Optional[str] = None):
    """记录LLM交互信息"""
    logger = get_logger("LLM交互")
    if error:
        logger.error(f"LLM {action} 失败: {error}")
        if details:
            logger.debug(f"详细信息: {details}")
    else:
        logger.info(f"LLM {action}")
        if details:
            logger.debug(f"详细信息: {details}")

def log_agent_event(event_type: str, content: str, llm_response_id: Optional[str] = None):
    """记录Agent事件"""
    logger = get_logger("Agent事件")
    if llm_response_id:
        logger.debug(f"[{event_type}] ({llm_response_id}) {content}")
    else:
        logger.debug(f"[{event_type}] {content}")

def log_tool_call_parsing(success: bool, tool_name: str, args: Dict[str, Any], error: Optional[str] = None):
    """记录工具调用解析"""
    logger = get_logger("工具解析")
    if success:
        logger.debug(f"工具调用解析成功: {tool_name}({args})")
    else:
        logger.warning(f"工具调用解析失败: {tool_name}，错误: {error}")

def log_schema_generation(tool_name: str, schema: Dict[str, Any]):
    """记录Schema生成"""
    logger = get_logger("Schema生成")
    logger.debug(f"为工具 '{tool_name}' 生成Schema")
    logger.trace(f"Schema内容: {schema}")

def log_http_request(method: str, url: str, status_code: Optional[int] = None, error: Optional[str] = None):
    """记录HTTP请求"""
    logger = get_logger("HTTP请求")
    if error:
        logger.error(f"{method} {url} 失败: {error}")
    else:
        logger.info(f"{method} {url} 响应码: {status_code}")

def log_agent_completion(reason: str, iterations_used: int, max_iterations: Optional[int] = None):
    """记录Agent完成信息"""
    logger = get_logger("智能体")
    if max_iterations and max_iterations > 0:
        logger.info(f"Agent执行完成，原因: {reason}，使用迭代: {iterations_used}/{max_iterations}")
    else:
        logger.info(f"Agent执行完成，原因: {reason}，使用迭代: {iterations_used}")

# 默认设置日志器
setup_logger()