import threading
import time
import logging
from typing import Dict, Any
from langchain.tools import Tool
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.tools import tool

logger = logging.getLogger("SearxTool")
logger.setLevel(logging.INFO)


# 连接池实现（保持不变）
class SearxConnectionPool:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._pool = []
                cls._instance._max_size = 15
                cls._instance._active = 0
        return cls._instance

    def get_connection(self):
        with self._lock:
            if self._pool:
                return self._pool.pop()
            if self._active < self._max_size:
                self._active += 1
                logger.info(f"创建新Searx连接 | 活跃: {self._active}/{self._max_size}")
                return SearxSearchWrapper(
                    searx_host="http://localhost:8888",
                    k=3  # 只保留允许的参数
                )
            # 等待空闲连接（生产环境应加超时）
            logger.warning("Searx连接池已满，等待中...")
            while not self._pool:
                time.sleep(0.1)
            return self._pool.pop()

    def return_connection(self, conn):
        with self._lock:
            self._pool.append(conn)


# 用户级速率限制器（保持不变）
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
            # 清理过期记录
            self.user_limits[user_id] = [
                t for t in self.user_limits[user_id]
                if now - t < self.period
            ]
            if len(self.user_limits[user_id]) >= self.max_calls:
                logger.warning(f"用户 {user_id} 搜索请求超限")
                return "搜索请求过于频繁，请1分钟后重试"
            self.user_limits[user_id].append(now)
        return func(*args, **kwargs)


# 修复后的搜索执行函数
def _execute_search(query: str) -> str:
    pool = SearxConnectionPool()
    conn = None
    try:
        conn = pool.get_connection()
        # 关键修复：将 safesearch 和 timeout 作为 run() 方法的参数传递
        return conn.run(
            query=query,
            language="zh",
            region="zh-CN",
            engines=["baidu"],
            safesearch=1,  # 现在应该作为 run() 的参数
            timeout=5.0  # 超时设置也作为 run() 参数
        )
    except Exception as e:
        logger.error(f"Searx搜索失败: {str(e)}")
        return "网络搜索服务暂时不可用"
    finally:
        if conn:
            pool.return_connection(conn)


# 修复后的工具定义
@tool("web_search_tool")
def web_search_tool(query: str, config: dict = None) -> str:
    """
    实时中文网络搜索工具（百度引擎），返回前3条结果摘要
    Args:
        query: 需要搜索的中文问题
        config: LangChain 配置对象
    Returns:
        str: 搜索结果摘要
    """
    import uuid

    # 从config中获取user_id
    user_id = "default_user"
    if config and isinstance(config, dict):
        user_id = config.get("configurable", {}).get("user_id", f"anon_{uuid.uuid4().hex}")
    else:
        user_id = f"anon_{uuid.uuid4().hex}"

    # 创建速率限制器
    rate_limiter = UserRateLimiter(max_calls=25, period=60)
    # 执行带速率限制的搜索
    return rate_limiter(user_id, _execute_search, query)


def create_search_tool():
    """返回符合LangGraph要求的搜索工具"""
    return web_search_tool