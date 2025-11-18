import os
import threading
import time
import logging
from typing import Dict, Any
from langchain.tools import Tool
from langchain_core.tools import tool
from tavily import TavilyClient  # 新增导入

logger = logging.getLogger("TavilyTool")
logger.setLevel(logging.INFO)


# 替换连接池实现（关键修改）
class TavilyConnectionPool:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._pool = []
                cls._instance._max_size = 15
                cls._instance._active = 0
                # 创建 Tavily 客户端（单例）
                api_key = "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"
                if not api_key:
                    raise RuntimeError("TAVILY_API_KEY 环境变量未设置")
                cls._instance._client = TavilyClient(api_key=api_key)
        return cls._instance

    def get_connection(self):
        with self._lock:
            if self._pool:
                return self._pool.pop()
            if self._active < self._max_size:
                self._active += 1
                logger.info(f"获取 Tavily 连接 | 活跃: {self._active}/{self._max_size}")
                return self._client  # 返回单例客户端
            logger.warning("Tavily 连接池已满，等待中...")
            while not self._pool:
                time.sleep(0.1)
            return self._pool.pop()

    def return_connection(self, conn):
        with self._lock:
            self._pool.append(conn)


# 保持用户级速率限制器不变（可选调整）
class UserRateLimiter:
    def __init__(self, max_calls=25, period=60):
        self.max_calls = max_calls
        self.period = period
        self.user_limits = {}
        self.lock = threading.Lock()

    def __call__(self, user_id, func, *args, **kwargs):
        with self.lock:
            now = time.time()
            if user_id not in self.user_limits:
                self.user_limits[user_id] = []
            self.user_limits[user_id] = [
                t for t in self.user_limits[user_id]
                if now - t < self.period
            ]
            if len(self.user_limits[user_id]) >= self.max_calls:
                logger.warning(f"用户 {user_id} 搜索请求超限")
                return "搜索请求过于频繁，请1分钟后重试"
            self.user_limits[user_id].append(now)
        return func(*args, **kwargs)


# 修改搜索执行函数（关键适配）
def _execute_search(query: str) -> str:
    pool = TavilyConnectionPool()
    client = None
    try:
        client = pool.get_connection()
        # 调用 Tavily API（适配参数）
        result = client.search(
            query=query,
            max_results=3,  # 对应原来的 k=3
            search_depth="basic",  # 基础搜索（可选 advanced）
            include_answer=True,  # 包含摘要答案
            include_raw_content=False
        )

        # 格式化结果（保持原有输出结构）
        if result.get("results"):
            formatted = []
            for i, item in enumerate(result["results"]):
                formatted.append(
                    f"{i + 1}. [{item['title']}]({item['url']})\n{item['content']}"
                )
            return "\n\n".join(formatted)
        return "未找到相关结果"
    except Exception as e:
        logger.error(f"Tavily 搜索失败: {str(e)}")
        return "网络搜索服务暂时不可用"
    finally:
        if client:
            pool.return_connection(client)


# 保持工具定义不变
@tool("web_search_tool")
def web_search_tool(query: str, config: dict = None) -> str:
    """
    实时中文网络搜索工具（Tavily API），返回前3条结果摘要
    Args:
        query: 需要搜索的中文问题
        config: LangChain 配置对象
    Returns:
        str: 搜索结果摘要
    """
    import uuid

    user_id = "default_user"
    if config and isinstance(config, dict):
        user_id = config.get("configurable", {}).get("user_id", f"anon_{uuid.uuid4().hex}")
    else:
        user_id = f"anon_{uuid.uuid4().hex}"

    rate_limiter = UserRateLimiter(max_calls=25, period=60)
    return rate_limiter(user_id, _execute_search, query)


def create_search_tool():
    """返回符合LangGraph要求的搜索工具"""
    return web_search_tool