import threading
import time
import logging
import random
from typing import Dict, Any, Optional, List
from langchain_core.tools import tool
from langchain_community.utilities import SearxSearchWrapper
from tavily import TavilyClient

logger = logging.getLogger("SmartSearchTool")
logger.setLevel(logging.INFO)

class SmartSearchEngine:
    """智能搜索引擎，支持多后端故障转移和负载均衡"""
    
    def __init__(self):
        self.searx_config = {
            "host": "http://localhost:8888",
            "k": 3,
            "language": "zh",
            "region": "zh-CN",
            "engines": ["baidu"],
            "safesearch": 1,
            "timeout": 5.0
        }
        
        self.tavily_config = {
            "api_key": "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",
            "max_results": 3,
            "search_depth": "basic",
            "include_answer": True,
            "include_raw_content": False
        }
        
        # 后端状态监控
        self.backend_status = {
            "searx": {"healthy": True, "last_check": time.time(), "error_count": 0, "total_calls": 0},
            "tavily": {"healthy": True, "last_check": time.time(), "error_count": 0, "total_calls": 0}
        }
        
        # 性能统计
        self.performance_stats = {
            "searx": {"total_calls": 0, "success_calls": 0, "avg_response_time": 0},
            "tavily": {"total_calls": 0, "success_calls": 0, "avg_response_time": 0}
        }
        
        # 连接池
        self.searx_pool = SearxConnectionPool()
        self.tavily_pool = TavilyConnectionPool()
        
        # 健康检查配置 - 按需执行，不定期执行
        self.health_check_interval = 300  # 5分钟，仅作为最大间隔
        self.last_health_check = time.time()
        
        logger.info("智能搜索引擎已初始化，健康检查将按需执行")
    
    def _check_backend_health(self):
        """检查后端健康状态 - 按需执行"""
        current_time = time.time()
        
        # 检查 SearXNG
        try:
            conn = self.searx_pool.get_connection()
            # 简单测试查询
            test_result = conn.run(
                query="test",
                language="zh",
                region="zh-CN",
                engines=["baidu"],
                safesearch=1,
                timeout=3.0
            )
            
            # 验证返回结果是否有效
            if test_result and isinstance(test_result, str) and len(test_result.strip()) > 10:
                self.backend_status["searx"]["healthy"] = True
                self.backend_status["searx"]["error_count"] = 0
                self.backend_status["searx"]["last_check"] = current_time
                logger.debug("SearXNG 健康检查通过")
            else:
                # 返回结果无效（空结果或格式错误）
                self.backend_status["searx"]["healthy"] = False
                self.backend_status["searx"]["error_count"] += 1
                self.backend_status["searx"]["last_check"] = current_time
                logger.warning(f"SearXNG 健康检查失败: 返回结果无效 - {test_result}")
                
        except Exception as e:
            self.backend_status["searx"]["healthy"] = False
            self.backend_status["searx"]["error_count"] += 1
            self.backend_status["searx"]["last_check"] = current_time
            logger.warning(f"SearXNG 健康检查失败: {str(e)}")
        finally:
            # 确保连接被正确返回
            try:
                if 'conn' in locals():
                    self.searx_pool.return_connection(conn)
            except:
                pass
        
        # 检查 Tavily
        try:
            client = self.tavily_pool.get_connection()
            # 简单测试查询
            test_result = client.search(
                query="test",
                max_results=1,
                search_depth="basic"
            )
            
            # 验证返回结果是否有效
            if test_result and isinstance(test_result, dict) and test_result.get("results"):
                self.backend_status["tavily"]["healthy"] = True
                self.backend_status["tavily"]["error_count"] = 0
                self.backend_status["tavily"]["last_check"] = current_time
                logger.debug("Tavily 健康检查通过")
            else:
                # 返回结果无效
                self.backend_status["tavily"]["healthy"] = False
                self.backend_status["tavily"]["error_count"] += 1
                self.backend_status["tavily"]["last_check"] = current_time
                logger.warning(f"Tavily 健康检查失败: 返回结果无效 - {test_result}")
                
        except Exception as e:
            self.backend_status["tavily"]["healthy"] = False
            self.backend_status["tavily"]["error_count"] += 1
            self.backend_status["tavily"]["last_check"] = current_time
            logger.warning(f"Tavily 健康检查失败: {str(e)}")
        finally:
            # 确保连接被正确返回
            try:
                if 'client' in locals():
                    self.tavily_pool.return_connection(client)
            except:
                pass
        
        self.last_health_check = current_time
        logger.info("后端健康检查完成")
    
    def _should_check_health(self) -> bool:
        """判断是否应该执行健康检查"""
        current_time = time.time()
        
        # 如果从未检查过，需要检查
        if self.last_health_check == 0:
            return True
        
        # 如果距离上次检查超过最大间隔，需要检查
        if current_time - self.last_health_check > self.health_check_interval:
            return True
        
        # 如果所有后端都标记为不健康，需要检查
        if not any(status["healthy"] for status in self.backend_status.values()):
            return True
        
        # 如果某个后端错误次数过多，需要检查
        for backend, status in self.backend_status.items():
            if status["error_count"] >= 3:  # 连续3次错误后强制检查
                return True
        
        return False
    
    def _select_backend(self, query: str) -> str:
        """智能选择搜索后端"""
        # 按需执行健康检查
        if self._should_check_health():
            logger.info("执行按需健康检查")
            self._check_backend_health()
        
        # 获取可用后端
        available_backends = []
        for backend, status in self.backend_status.items():
            if status["healthy"]:
                available_backends.append(backend)
        
        if not available_backends:
            logger.error("所有搜索后端都不可用")
            # 强制重新检查
            self._check_backend_health()
            # 再次检查可用性
            available_backends = [b for b, s in self.backend_status.items() if s["healthy"]]
            if not available_backends:
                raise RuntimeError("所有搜索后端都不可用")
        
        # 如果只有一个可用后端，直接使用
        if len(available_backends) == 1:
            selected = available_backends[0]
            logger.info(f"只有一个可用后端: {selected}")
            return selected
        
        # 多个后端可用时，使用智能选择策略
        # 策略1: 根据错误率选择
        error_rates = {}
        for backend in available_backends:
            status = self.backend_status[backend]
            if status["total_calls"] > 0:
                error_rate = status["error_count"] / status["total_calls"]
            else:
                error_rate = 0
            error_rates[backend] = error_rate
        
        # 策略2: 根据响应时间选择
        response_times = {}
        for backend in available_backends:
            stats = self.performance_stats[backend]
            response_times[backend] = stats["avg_response_time"]
        
        # 综合评分：错误率权重0.7，响应时间权重0.3
        scores = {}
        for backend in available_backends:
            error_score = 1 - error_rates[backend]  # 错误率越低分数越高
            time_score = 1 / (1 + response_times[backend])  # 响应时间越短分数越高
            scores[backend] = 0.7 * error_score + 0.3 * time_score
        
        # 选择最高分后端
        selected = max(scores.items(), key=lambda x: x[1])[0]
        logger.info(f"智能选择后端: {selected} (分数: {scores[selected]:.3f})")
        
        return selected
    
    def search(self, query: str) -> str:
        """执行智能搜索"""
        start_time = time.time()
        backend = None
        
        try:
            # 选择后端
            backend = self._select_backend(query)
            
            # 设置超时检测
            max_timeout = self.searx_config["timeout"] if backend == "searx" else 10.0
            
            # 使用线程执行搜索，支持超时检测
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                if backend == "searx":
                    future = executor.submit(self._search_searx, query)
                else:
                    future = executor.submit(self._search_tavily, query)
                
                try:
                    result = future.result(timeout=max_timeout)
                except concurrent.futures.TimeoutError:
                    raise ValueError(f"后端 {backend} 搜索超时 (>{max_timeout}s)")
            
            # 验证搜索结果
            if not result or len(str(result).strip()) < 10:
                raise ValueError(f"后端 {backend} 返回结果无效: {result}")
            
            # 检查是否包含错误信息
            error_indicators = ["error", "Error", "ERROR", "failed", "Failed", "FAILED", "timeout", "Timeout", "TIMEOUT"]
            if any(indicator in str(result) for indicator in error_indicators):
                raise ValueError(f"后端 {backend} 返回错误信息: {result[:100]}")
            
            # 更新性能统计
            response_time = time.time() - start_time
            self._update_performance_stats(backend, True, response_time)
            
            # 添加来源标记
            result_with_source = f"[{backend.upper()}] {result}"
            logger.info(f"搜索完成 - 后端: {backend}, 响应时间: {response_time:.2f}s")
            
            return result_with_source
            
        except Exception as e:
            # 更新性能统计
            response_time = time.time() - start_time
            if backend:
                self._update_performance_stats(backend, False, response_time)
            
            # 尝试故障转移
            if backend:
                logger.warning(f"后端 {backend} 搜索失败，尝试故障转移: {str(e)}")
                return self._fallback_search(query, backend)
            else:
                logger.error(f"无法选择后端，搜索失败: {str(e)}")
                return f"搜索服务暂时不可用: {str(e)}"
    
    def _search_searx(self, query: str) -> str:
        """使用 SearXNG 搜索"""
        conn = None
        try:
            conn = self.searx_pool.get_connection()
            result = conn.run(
                query=query,
                language=self.searx_config["language"],
                region=self.searx_config["region"],
                engines=self.searx_config["engines"],
                safesearch=self.searx_config["safesearch"],
                timeout=self.searx_config["timeout"]
            )
            
            # 验证搜索结果
            if not result:
                raise ValueError("SearXNG 返回空结果")
            
            if not isinstance(result, str):
                raise ValueError(f"SearXNG 返回类型错误: {type(result)}")
            
            if len(result.strip()) < 10:
                raise ValueError(f"SearXNG 返回结果过短: {result}")
            
            # 检查是否包含错误信息
            error_indicators = ["error", "Error", "ERROR", "failed", "Failed", "FAILED", "timeout", "Timeout", "TIMEOUT"]
            if any(indicator in result for indicator in error_indicators):
                raise ValueError(f"SearXNG 返回错误信息: {result[:100]}")
            
            return result
            
        except Exception as e:
            logger.error(f"SearXNG 搜索失败: {str(e)}")
            raise
        finally:
            if conn:
                try:
                    self.searx_pool.return_connection(conn)
                except Exception as e:
                    logger.warning(f"返回 SearXNG 连接到池失败: {str(e)}")
    
    def _search_tavily(self, query: str) -> str:
        """使用 Tavily 搜索"""
        client = None
        try:
            client = self.tavily_pool.get_connection()
            result = client.search(
                query=query,
                max_results=self.tavily_config["max_results"],
                search_depth=self.tavily_config["search_depth"],
                include_answer=self.tavily_config["include_answer"],
                include_raw_content=self.tavily_config["include_raw_content"]
            )
            
            # 验证搜索结果
            if not result:
                raise ValueError("Tavily 返回空结果")
            
            if not isinstance(result, dict):
                raise ValueError(f"Tavily 返回类型错误: {type(result)}")
            
            # 检查是否包含错误信息
            if "error" in result or "Error" in result:
                raise ValueError(f"Tavily 返回错误: {result}")
            
            # 格式化结果（保持原有输出结构）
            if result.get("results") and len(result["results"]) > 0:
                formatted = []
                for i, item in enumerate(result["results"]):
                    # 验证每个结果项
                    if not item.get("title") or not item.get("content"):
                        continue
                    
                    title = item["title"].strip()
                    content = item["content"].strip()
                    url = item.get("url", "")
                    
                    if title and content:
                        formatted.append(
                            f"{i + 1}. [{title}]({url})\n{content}"
                        )
                
                if formatted:
                    return "\n\n".join(formatted)
                else:
                    raise ValueError("Tavily 结果格式化后为空")
            else:
                raise ValueError("Tavily 未返回有效结果")
                
        except Exception as e:
            logger.error(f"Tavily 搜索失败: {str(e)}")
            raise
        finally:
            if client:
                try:
                    self.tavily_pool.return_connection(client)
                except Exception as e:
                    logger.warning(f"返回 Tavily 连接到池失败: {str(e)}")
    
    def _fallback_search(self, query: str, failed_backend: str) -> str:
        """故障转移搜索"""
        # 标记失败的后端
        self.backend_status[failed_backend]["healthy"] = False
        self.backend_status[failed_backend]["error_count"] += 1
        
        # 尝试其他后端
        other_backends = [b for b in self.backend_status.keys() if b != failed_backend]
        
        for backend in other_backends:
            try:
                logger.info(f"尝试故障转移到后端: {backend}")
                
                # 添加重试机制
                max_retries = 2
                for retry in range(max_retries):
                    try:
                        if backend == "searx":
                            result = self._search_searx(query)
                        else:
                            result = self._search_tavily(query)
                        
                        # 验证故障转移结果
                        if not result or len(str(result).strip()) < 10:
                            raise ValueError(f"故障转移结果无效: {result}")
                        
                        # 故障转移成功，恢复失败的后端状态
                        self.backend_status[failed_backend]["healthy"] = True
                        self.backend_status[failed_backend]["error_count"] = 0
                        logger.info(f"故障转移成功，使用后端: {backend}")
                        
                        return f"[{backend.upper()}-故障转移] {result}"
                        
                    except Exception as retry_error:
                        if retry < max_retries - 1:
                            logger.warning(f"故障转移到 {backend} 第{retry+1}次失败，重试中: {str(retry_error)}")
                            time.sleep(0.5)  # 短暂延迟后重试
                        else:
                            raise retry_error
                
            except Exception as e:
                logger.warning(f"故障转移到 {backend} 最终失败: {str(e)}")
                self.backend_status[backend]["healthy"] = False
                self.backend_status[backend]["error_count"] += 1
        
        # 如果所有后端都失败，触发健康检查
        if not any(status["healthy"] for status in self.backend_status.values()):
            logger.info("所有后端都不可用，触发健康检查")
            self._check_backend_health()
        
        # 再次尝试搜索
        for backend in self.backend_status.keys():
            if self.backend_status[backend]["healthy"]:
                try:
                    logger.info(f"后端 {backend} 已恢复，尝试搜索...")
                    if backend == "searx":
                        result = self._search_searx(query)
                    else:
                        result = self._search_tavily(query)
                    
                    if result and len(str(result).strip()) > 10:
                        logger.info(f"后端 {backend} 恢复搜索成功")
                        return f"[{backend.upper()}-恢复] {result}"
                        
                except Exception as e:
                    logger.warning(f"后端 {backend} 恢复搜索失败: {str(e)}")
        
        # 所有尝试都失败
        error_msg = f"所有搜索后端都不可用，请稍后重试。最后错误: {str(e) if 'e' in locals() else '未知错误'}"
        logger.error(error_msg)
        return error_msg
    
    def _update_performance_stats(self, backend: str, success: bool, response_time: float):
        """更新性能统计"""
        stats = self.performance_stats[backend]
        stats["total_calls"] += 1
        
        if success:
            stats["success_calls"] += 1
            # 更新平均响应时间
            if stats["success_calls"] == 1:
                stats["avg_response_time"] = response_time
            else:
                stats["avg_response_time"] = (
                    (stats["avg_response_time"] * (stats["success_calls"] - 1) + response_time) 
                    / stats["success_calls"]
                )
        
        # 更新后端状态统计
        if backend in self.backend_status:
            self.backend_status[backend]["total_calls"] = stats["total_calls"]
    
    def get_status(self) -> Dict[str, Any]:
        """获取搜索引擎状态"""
        return {
            "backend_status": self.backend_status,
            "performance_stats": self.performance_stats,
            "last_health_check": self.last_health_check,
            "health_check_interval": self.health_check_interval
        }
    
    def force_health_check(self) -> Dict[str, Any]:
        """手动强制执行健康检查"""
        logger.info("手动触发健康检查")
        self._check_backend_health()
        return {
            "status": "health_check_completed",
            "backend_status": self.backend_status,
            "timestamp": time.time()
        }

# 连接池实现
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
                    k=3
                )
            logger.warning("Searx连接池已满，等待中...")
            while not self._pool:
                time.sleep(0.1)
            return self._pool.pop()

    def return_connection(self, conn):
        with self._lock:
            self._pool.append(conn)

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
                cls._instance._client = TavilyClient(api_key="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")
        return cls._instance

    def get_connection(self):
        with self._lock:
            if self._pool:
                return self._pool.pop()
            if self._active < self._max_size:
                self._active += 1
                logger.info(f"获取 Tavily 连接 | 活跃: {self._active}/{self._max_size}")
                return self._client
            logger.warning("Tavily 连接池已满，等待中...")
            while not self._pool:
                time.sleep(0.1)
            return self._pool.pop()

    def return_connection(self, conn):
        with self._lock:
            self._pool.append(conn)

# 用户级速率限制器
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

# 创建全局智能搜索引擎实例
smart_search_engine = SmartSearchEngine()

@tool("web_search_tool")
def web_search_tool(query: str, config: dict = None) -> str:
    """
    智能网络搜索工具，自动在多个搜索后端之间切换，提供故障转移和负载均衡
    Args:
        query: 需要搜索的问题
        config: LangChain 配置对象
    Returns:
        str: 搜索结果摘要，包含来源标记
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
    return rate_limiter(user_id, smart_search_engine.search, query)

def create_search_tool():
    """返回符合LangGraph要求的搜索工具"""
    return web_search_tool

def get_search_status():
    """获取搜索引擎状态信息"""
    return smart_search_engine.get_status()

# 测试函数
if __name__ == "__main__":
    # 测试搜索功能
    test_queries = [
        "人工智能最新发展",
        "机器学习应用",
        "深度学习技术"
    ]
    
    print("=== 智能搜索测试 ===")
    for query in test_queries:
        print(f"\n查询: {query}")
        try:
            result = smart_search_engine.search(query)
            print(f"结果: {result[:200]}...")
        except Exception as e:
            print(f"错误: {str(e)}")
    
    # 显示状态
    print(f"\n=== 搜索引擎状态 ===")
    status = smart_search_engine.get_status()
    for backend, info in status["backend_status"].items():
        print(f"{backend}: {'健康' if info['healthy'] else '异常'} (错误次数: {info['error_count']})")
    
    for backend, stats in status["performance_stats"].items():
        if stats['total_calls'] > 0:
            success_rate = (stats['success_calls'] / stats['total_calls']) * 100
            print(f"{backend} 性能: 总调用 {stats['total_calls']}, 成功率 {success_rate:.1f}%")
        else:
            print(f"{backend} 性能: 暂无调用数据")
