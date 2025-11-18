"""
NSRL DeepSeek API 自定义客户端
用于处理 NSRL API 的特殊路径 /portal/api/ask
"""
import httpx
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.tools import BaseTool
import logging

logger = logging.getLogger(__name__)


class NSRLDeepSeekChat(BaseChatModel):
    """NSRL DeepSeek API 自定义客户端"""
    
    api_key: str
    api_base: str = "http://scc.ustc.edu.cn/portal/api/ask"
    model: str = "deepseek-v3"
    max_tokens: int = 10000
    temperature: float = 0.1
    request_timeout: float = 120.0
    max_retries: int = 5
    
    @property
    def _llm_type(self) -> str:
        return "nsrl-deepseek"
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """流式生成（暂不支持）"""
        # NSRL API 可能不支持流式，先返回非流式结果
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        yield result.generations[0]
    
    def bind_tools(
        self,
        tools: Union[List[Dict[str, Any]], List[BaseTool], List[type]],
        **kwargs: Any,
    ) -> "NSRLDeepSeekChat":
        """
        绑定工具到模型（返回一个新的实例，支持工具调用）
        
        注意：NSRL API 可能支持工具调用，我们需要将工具转换为函数定义格式
        """
        # 转换工具为函数定义格式
        functions = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                # 从 BaseTool 提取函数定义
                func_def = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": self._extract_tool_schema(tool)
                    }
                }
                functions.append(func_def)
            elif isinstance(tool, dict):
                # 已经是字典格式，直接使用
                functions.append(tool)
            else:
                logger.warning(f"未知的工具类型: {type(tool)}, 跳过")
        
        # 创建一个新的实例，带有工具绑定
        bound_instance = NSRLDeepSeekChat(
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            request_timeout=self.request_timeout,
            max_retries=self.max_retries,
        )
        # 存储绑定的工具
        bound_instance._bound_tools = functions
        return bound_instance
    
    def _extract_tool_schema(self, tool: BaseTool) -> Dict[str, Any]:
        """从工具中提取 JSON Schema"""
        # 尝试从工具的 args_schema 获取
        if hasattr(tool, 'args_schema') and tool.args_schema:
            schema = tool.args_schema.schema()
            # 移除 title 等不需要的字段
            return {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        else:
            # 如果没有 schema，返回空 schema
            return {
                "type": "object",
                "properties": {},
                "required": []
            }
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成回复"""
        # 转换消息格式
        api_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                api_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
        
        # 构建请求数据
        payload = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        
        # 如果绑定了工具，添加到请求中
        if hasattr(self, '_bound_tools') and self._bound_tools:
            payload["tools"] = self._bound_tools
            # DeepSeek API 使用 tool_choice 参数控制工具调用
            # "auto" 表示模型可以自主选择是否调用工具
            payload["tool_choice"] = "auto"
        
        # 添加 stop 参数（如果提供）
        if stop:
            payload["stop"] = stop
        
        # 合并额外的 kwargs
        payload.update(kwargs)
        
        # 发送请求
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # 明确指定 Accept-Encoding，让 httpx 正确处理 gzip
            "Accept-Encoding": "gzip, deflate",
        }
        
        logger.info(f"发送请求到: {self.api_base}")
        logger.debug(f"请求数据: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        
        # 使用 httpx 发送请求（同步版本，用于同步调用）
        try:
            with httpx.Client(timeout=self.request_timeout) as client:
                response = client.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                
                # 检查响应内容类型
                content_type = response.headers.get("content-type", "")
                content_encoding = response.headers.get("content-encoding", "")
                logger.info(f"响应 Content-Type: {content_type}")
                logger.info(f"响应 Content-Encoding: {content_encoding}")
                
                # 检查响应是否为空
                # 注意：即使响应被 gzip 压缩，httpx 应该自动解压，response.text 应该是解压后的内容
                response_text = response.text
                if not response_text or len(response_text.strip()) == 0:
                    logger.error(f"响应为空！状态码: {response.status_code}")
                    logger.error(f"响应头: {dict(response.headers)}")
                    # 尝试读取原始内容（如果可能）
                    try:
                        raw_content = response.content
                        logger.error(f"原始响应内容长度: {len(raw_content)}")
                        if len(raw_content) > 0:
                            logger.error(f"原始响应内容 (前100字符): {raw_content[:100]}")
                    except:
                        pass
                    raise ValueError("API返回了空响应")
                
                # 检查 Content-Type，如果不是 JSON，记录详细信息
                if "application/json" not in content_type.lower():
                    response_text = response.text[:1000]
                    logger.error(f"API返回了非JSON格式！")
                    logger.error(f"Content-Type: {content_type}")
                    logger.error(f"响应状态码: {response.status_code}")
                    logger.error(f"响应长度: {len(response.text)}")
                    logger.error(f"响应内容 (前1000字符): {response_text}")
                    # 如果是 HTML，可能是错误页面
                    if "text/html" in content_type.lower():
                        raise ValueError(f"API返回了HTML格式的响应（可能是错误页面），Content-Type: {content_type}")
                    else:
                        raise ValueError(f"API返回了非JSON格式的响应，Content-Type: {content_type}")
                
                # 尝试解析 JSON
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    # 如果 JSON 解析失败，记录完整的响应信息
                    response_text = response.text[:1000]  # 记录前1000字符
                    logger.error(f"JSON解析失败: {str(e)}")
                    logger.error(f"响应状态码: {response.status_code}")
                    logger.error(f"响应 Content-Type: {content_type}")
                    logger.error(f"响应长度: {len(response.text)}")
                    logger.error(f"响应内容 (前1000字符): {response_text}")
                    logger.error(f"响应是否以{{开头: {response.text.strip().startswith('{')}")
                    logger.error(f"响应是否包含HTML: {'<html' in response.text.lower() or '<!DOCTYPE' in response.text}")
                    raise ValueError(f"API返回的不是有效的JSON格式: {str(e)}")
                    
        except httpx.TimeoutException:
            logger.error(f"请求超时: {self.api_base}")
            raise
        except httpx.HTTPStatusError as e:
            error_text = e.response.text[:500] if e.response.text else "无响应内容"
            logger.error(f"HTTP错误: {e.response.status_code}")
            logger.error(f"错误响应内容 (前500字符): {error_text}")
            raise
        except Exception as e:
            logger.error(f"请求失败: {type(e).__name__}: {str(e)}")
            raise
        
        logger.debug(f"响应数据: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        # 解析响应
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            message = choice.get("message", {})
            message_content = message.get("content")
            
            # 处理content为None的情况
            if message_content is None:
                logger.warning("[同步] API返回的content为None，使用空字符串代替")
                message_content = ""
            
            # 检查是否有工具调用
            tool_calls = message.get("tool_calls", [])
            
            # 创建 AIMessage，包含工具调用信息
            ai_message = AIMessage(content=message_content)
            if tool_calls:
                # 如果有工具调用，添加到消息中
                # LangChain 使用 ToolMessage 来处理工具调用
                for tool_call in tool_calls:
                    if "function" in tool_call:
                        func_name = tool_call["function"].get("name", "")
                        func_args = tool_call["function"].get("arguments", "{}")
                        # 解析 JSON 参数
                        try:
                            func_args_dict = json.loads(func_args)
                        except:
                            func_args_dict = {}
                        
                        # 添加工具调用信息到消息中
                        # LangChain 的 AIMessage 支持 tool_calls 属性
                        if not hasattr(ai_message, "tool_calls"):
                            ai_message.tool_calls = []
                        ai_message.tool_calls.append({
                            "name": func_name,
                            "args": func_args_dict,
                            "id": tool_call.get("id", "")
                        })
            
            # 创建 ChatGeneration
            generation = ChatGeneration(
                message=ai_message,
                generation_info=result.get("usage", {}),
            )
            
            return ChatResult(generations=[generation])
        else:
            raise ValueError(f"意外的响应格式: {result}")
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步生成回复"""
        # 转换消息格式
        api_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                api_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
        
        # 构建请求数据
        payload = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        
        # 如果绑定了工具，添加到请求中
        if hasattr(self, '_bound_tools') and self._bound_tools:
            payload["tools"] = self._bound_tools
            payload["tool_choice"] = "auto"
        
        # 添加 stop 参数（如果提供）
        if stop:
            payload["stop"] = stop
        
        # 合并额外的 kwargs
        payload.update(kwargs)
        
        # 发送请求
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # 明确指定 Accept-Encoding，让 httpx 正确处理 gzip
            "Accept-Encoding": "gzip, deflate",
        }
        
        logger.info(f"[异步] 发送请求到: {self.api_base}")
        logger.debug(f"[异步] 请求数据: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        
        # 使用异步 httpx 客户端发送请求
        try:
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                response = await client.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                
                # 检查响应内容类型
                content_type = response.headers.get("content-type", "")
                content_encoding = response.headers.get("content-encoding", "")
                logger.info(f"[异步] 响应 Content-Type: {content_type}")
                logger.info(f"[异步] 响应 Content-Encoding: {content_encoding}")
                
                # 检查响应是否为空
                # 注意：即使响应被 gzip 压缩，httpx 应该自动解压，response.text 应该是解压后的内容
                response_text = response.text
                if not response_text or len(response_text.strip()) == 0:
                    logger.error(f"[异步] 响应为空！状态码: {response.status_code}")
                    logger.error(f"[异步] 响应头: {dict(response.headers)}")
                    # 尝试读取原始内容（如果可能）
                    try:
                        raw_content = response.content
                        logger.error(f"[异步] 原始响应内容长度: {len(raw_content)}")
                        if len(raw_content) > 0:
                            logger.error(f"[异步] 原始响应内容 (前100字符): {raw_content[:100]}")
                    except:
                        pass
                    raise ValueError("API返回了空响应")
                
                # 检查 Content-Type，如果不是 JSON，记录详细信息
                if "application/json" not in content_type.lower():
                    response_text = response.text[:1000]
                    logger.error(f"[异步] API返回了非JSON格式！")
                    logger.error(f"[异步] Content-Type: {content_type}")
                    logger.error(f"[异步] 响应状态码: {response.status_code}")
                    logger.error(f"[异步] 响应长度: {len(response.text)}")
                    logger.error(f"[异步] 响应内容 (前1000字符): {response_text}")
                    # 如果是 HTML，可能是错误页面
                    if "text/html" in content_type.lower():
                        raise ValueError(f"API返回了HTML格式的响应（可能是错误页面），Content-Type: {content_type}")
                    else:
                        raise ValueError(f"API返回了非JSON格式的响应，Content-Type: {content_type}")
                
                # 尝试解析 JSON
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    # 如果 JSON 解析失败，记录完整的响应信息
                    response_text = response.text[:1000]  # 记录前1000字符
                    logger.error(f"[异步] JSON解析失败: {str(e)}")
                    logger.error(f"[异步] 响应状态码: {response.status_code}")
                    logger.error(f"[异步] 响应 Content-Type: {content_type}")
                    logger.error(f"[异步] 响应长度: {len(response.text)}")
                    logger.error(f"[异步] 响应内容 (前1000字符): {response_text}")
                    logger.error(f"[异步] 响应是否以{{开头: {response.text.strip().startswith('{')}")
                    logger.error(f"[异步] 响应是否包含HTML: {'<html' in response.text.lower() or '<!DOCTYPE' in response.text}")
                    raise ValueError(f"API返回的不是有效的JSON格式: {str(e)}")
                    
        except httpx.TimeoutException:
            logger.error(f"[异步] 请求超时: {self.api_base}")
            raise
        except httpx.HTTPStatusError as e:
            error_text = e.response.text[:500] if e.response.text else "无响应内容"
            logger.error(f"[异步] HTTP错误: {e.response.status_code}")
            logger.error(f"[异步] 错误响应内容 (前500字符): {error_text}")
            raise
        except Exception as e:
            logger.error(f"[异步] 请求失败: {type(e).__name__}: {str(e)}")
            raise
        
        logger.debug(f"[异步] 响应数据: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        # 解析响应
        if "choices" in result and len(result["choices"]) > 0:
            choice = result["choices"][0]
            message = choice.get("message", {})
            message_content = message.get("content")
            
            # 处理content为None的情况
            if message_content is None:
                logger.warning("[异步] API返回的content为None，使用空字符串代替")
                message_content = ""
            
            # 检查是否有工具调用
            tool_calls = message.get("tool_calls", [])
            
            # 创建 AIMessage，包含工具调用信息
            ai_message = AIMessage(content=message_content)
            if tool_calls:
                for tool_call in tool_calls:
                    if "function" in tool_call:
                        func_name = tool_call["function"].get("name", "")
                        func_args = tool_call["function"].get("arguments", "{}")
                        try:
                            func_args_dict = json.loads(func_args)
                        except:
                            func_args_dict = {}
                        
                        if not hasattr(ai_message, "tool_calls"):
                            ai_message.tool_calls = []
                        ai_message.tool_calls.append({
                            "name": func_name,
                            "args": func_args_dict,
                            "id": tool_call.get("id", "")
                        })
            
            # 创建 ChatGeneration
            generation = ChatGeneration(
                message=ai_message,
                generation_info=result.get("usage", {}),
            )
            
            logger.info(f"[异步] 请求成功，收到响应")
            return ChatResult(generations=[generation])
        else:
            raise ValueError(f"意外的响应格式: {result}")

