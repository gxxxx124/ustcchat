import operator
import re
from email._header_value_parser import parse_message_id

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, AsyncGenerator
import uuid
import logging
from contextlib import asynccontextmanager
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict
from operator import add
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import os
import asyncio
from fastapi.responses import StreamingResponse
from psycopg_pool import AsyncConnectionPool
import json
# 缓存已创建的RAG工具实例，避免重复创建
rag_tool_cache = {}


def get_rag_tool(knowledge_base_name: str):
    """根据知识库名称获取或创建RAG工具"""
    if knowledge_base_name in rag_tool_cache:
        return rag_tool_cache[knowledge_base_name]

    # 创建新的RAG工具实例
    from rag_tool import create_rag_tool
    rag_tool = create_rag_tool(
        host="localhost",
        port=6333,
        collection_name=knowledge_base_name
    )
    rag_tool_cache[knowledge_base_name] = rag_tool
    return rag_tool

from pdf import (
    register_user_document_tool,
    get_user_document_tool,
    get_user_document_tool_by_session,
    list_user_document_tools
)
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("agent-api")

# 1. 配置模型
inference_server_url = "http://localhost:11434/v1"
model = ChatOpenAI(
    model="qwen3:4b",
    openai_api_key="none",
    openai_api_base=inference_server_url,
    max_tokens=500,
    temperature=0.7,  # 降低温度提高决策稳定性
    timeout=30.0  # 添加超时防止卡死
)


def extract_highest_similarity(tool_response: str) -> float:
    """从工具响应中提取最高相似度"""
    # 使用正则表达式查找所有相似度值
    similarity_values = re.findall(r"相似度: ([\d.]+)", tool_response)
    if not similarity_values:
        logger.warning("未在工具响应中找到相似度信息")
        return 0.0
    # 转换为浮点数并返回最大值
    try:
        similarities = [float(val) for val in similarity_values]
        highest = max(similarities)
        logger.info(f"检测到最高相似度: {highest:.4f}")
        return highest
    except ValueError:
        logger.error("无法解析相似度值")
        return 0.0


# 2. 创建工具实例 - 注意：现在不创建固定的RAG工具，将在call_model中动态创建
try:
    from searxng_server import create_search_tool

    web_search_tool = create_search_tool()
    logger.info("✅ 工具加载成功: web_search")
except Exception as e:
    logger.error(f"❌ 工具加载失败: {str(e)}")
    raise


# 3. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add]
    tool_call_count: Annotated[int, operator.add]  # 添加工具调用计数器
    knowledge_base_name: str  # 新增字段，用于存储当前会话使用的知识库名称
    user_document_tools: List[str]  # 新增字段，用于存储当前会话可用的用户文档工具名称
    web_search_enabled: bool  # 新增：记录web搜索是否启用


# 4. 修改模型调用节点
async def call_model(state: AgentState):
    """模型自主决策是否需要调用工具，包含参数验证和状态更新"""
    messages = state["messages"]
    knowledge_base_name = state.get("knowledge_base_name", "test")

    # 动态获取当前知识库的RAG工具
    rag_tool = get_rag_tool(knowledge_base_name)
    available_tools = [rag_tool]
    if state.get("web_search_enabled", True):
        available_tools.append(web_search_tool)

    # 添加用户文档工具（如果有）
    user_document_tools_list = state.get("user_document_tools", [])
    for tool_name in user_document_tools_list:
        tool_info = get_user_document_tool(tool_name)
        if tool_info and "tool" in tool_info:
            available_tools.append(tool_info["tool"])
    # 如果是初始查询，添加系统提示
        # 如果是初始查询，添加系统提示
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            # 构建工具列表描述
            tools_description = f"""1. rag_knowledge_search: 查询内部知识库（主要包含医学相关内容）
               - 必须参数: query (string)
               - 当前知识库: {knowledge_base_name}
               - 调用示例: {{"name": "rag_knowledge_search", "arguments": {{"query": "NGS适用人群"}}}}
            2. web_search_tool: 查询最新互联网信息
               - 必须参数: query (string)
               - 调用示例: {{"name": "web_search_tool", "arguments": {{"query": "NGS最新研究"}}}}"""

            # 添加用户文档工具描述
            if user_document_tools_list:
                tools_description += "\n\n用户上传的文档搜索工具:"
                for tool_name in user_document_tools_list:
                    tool_info = get_user_document_tool(tool_name)
                    if tool_info:
                        tools_description += f"\n{tool_info['tool'].name}: {tool_info['tool'].description}"
                        tools_description += "\n   - 必须参数: query (string)"

            # 构建系统提示
            if not state.get("web_search_enabled", True):
                system_prompt = f"""你是一个大型粒子加速器专家，专门回答关于粒子加速器的技术问题。

可用工具：
{tools_description}

工作流程：
1. 对于所有粒子加速器相关问题，必须首先使用 rag_knowledge_search 搜索知识库
2. 检查返回结果的最高相似度：
   * 如果最高相似度 ≥ 0.6，结果相关，基于此生成回答
   * 如果最高相似度 < 0.6，结果不相关，明确告知用户知识库中没有相关信息
3. 严格基于知识库内容回答，不得编造或推测信息

重要指导原则:
1. 回答必须严格基于知识库内容，不得编造信息
2. 如果知识库中没有相关信息，明确告知用户"知识库中没有找到相关信息"
3. 提供信息时要注明来源（来自知识库）
4. 对于技术建议，必须基于知识库中的权威资料
5. 如果知识库搜索结果相似度 < 0.6，不得基于低相似度结果生成回答
6. 不得使用网络搜索或其他外部信息源
7. 回答要专业、准确、严谨
8. 如果已经调用工具超过3次仍未找到相关信息，明确告知用户知识库中无相关信息

    工具调用格式要求:
    - 仅使用指定的工具名称
    - 仅传递工具定义中要求的参数
    - 绝对不要添加额外参数（如"using"、"reason"等）
    - 严格按照JSON格式输出工具调用
    - 例如: {{"name": "rag_knowledge_search", "arguments": {{"query": "你的查询"}}}}
    - 重要: 不要在工具调用中包含任何额外文本、解释或<think>标签
    - 工具调用必须是纯JSON格式，不能有其他内容
    - 错误示例: {{"name": "rag_knowledge_search", "arguments": {{"query": "...", "using": "..."}}}}"""
            else:
                system_prompt = f"""你是一个医疗智能助手，可以使用以下工具：
    {tools_description}
    工作流程：
    - 如果用户询问与其上传文档相关的内容，优先使用对应的文档搜索工具
    - 对于一般医学问题，首先尝试使用 rag_knowledge_search
    - 检查返回结果的最高相似度：
      * 如果最高相似度 ≥ 0.5，结果可能相关，可基于此生成回答
      * 如果最高相似度 < 0.5，结果不相关，应使用 web_search_tool
    - 确保最终回答整合所有可用信息

    重要指导原则:
    1. 优先使用用户上传的文档工具（如果问题与文档内容相关）
    2. 其次使用本地知识库工具(rag_knowledge_search)
    3. 当本地知识库搜索结果的最高相似度 < 0.5 时，必须使用网络搜索工具(web_search_tool),不得直接生成回答
    4. 回答必须基于证据，不要编造信息
    5. 提供信息时要注明来源（来自用户文档、本地知识库或网络搜索）
    6. 对于医学建议，必须提醒用户咨询专业医生
    7. 如果网络搜索返回了明确的答案，则直接生成最终答案
    8. 如果已经调用工具超过5次仍未解决问题，请基于已有信息提供最佳答案
    9. 当用户提到"我的报告"、"我的基因检测"等类似表述时，优先使用用户上传的文档工具
    10. 如果最后工具调用没有结果返回，则直接回答用户问题

    工具调用格式要求:
    - 仅使用指定的工具名称
    - 仅传递工具定义中要求的参数
    - 绝对不要添加额外参数（如"using"、"reason"等）
    - 严格按照JSON格式输出工具调用
    - 例如: {{"name": "web_search_tool", "arguments": {{"query": "你的查询"}}}}
    - 重要: 不要在工具调用中包含任何额外文本、解释或<think>标签
    - 工具调用必须是纯JSON格式，不能有其他内容
    - 错误示例: {{"name": "web_search_tool", "arguments": {{"query": "...", "using": "..."}}}}"""
            messages = [SystemMessage(content=system_prompt)] + messages

        # 检查工具调用次数 - 如果超过限制，强制模型提供答案
        if state.get("tool_call_count", 0) >= 5:
            messages.append(SystemMessage(
                content="⚠️ 重要提示：您已经调用了多次工具但仍未能提供最终答案。"
                        "请基于已有信息立即提供完整回答，不要再调用工具。"
            ))

        # 计数web_search调用次数
        web_search_count = sum(1 for m in messages
                               if isinstance(m, AIMessage) and
                               m.tool_calls and
                               any(tc["name"] == "web_search_tool" for tc in m.tool_calls))

        # 检查是否已经调用过web_search但结果不理想
        if web_search_count >= 5:
            messages.append(SystemMessage(
                content="⚠️ 重要提示：您已经多次使用网络搜索但仍未提供最终答案。"
                        "请基于已有信息立即提供完整回答，不要再调用工具。"
            ))

        # === 关键新增：检查上一次工具调用结果的相似度 ===
        if len(messages) >= 3:
            last_tool_response = messages[-2]  # 上一条是工具响应
            if isinstance(last_tool_response, ToolMessage) and last_tool_response.content:
                # 从工具响应中提取最高相似度
                highest_similarity = extract_highest_similarity(last_tool_response.content)
                # 如果最高相似度低于阈值，添加系统提示强制使用web_search_tool
                if highest_similarity < 0.5:
                    logger.warning(f"检测到低相关性结果 (相似度: {highest_similarity:.4f})，强制使用网络搜索")
                    messages.append(SystemMessage(
                        content="⚠️ 重要提示：本地知识库搜索结果相关性较低（最高相似度: {:.4f}）。"
                                "请使用web_search_tool获取最新互联网信息。".format(highest_similarity)
                    ))

        # 始终绑定所有可用工具
        model_with_tools = model.bind_tools(available_tools)

        # 调用模型
        response = await model_with_tools.ainvoke(messages)

        # 关键修复：验证并清理工具调用参数
        if hasattr(response, "tool_calls") and response.tool_calls:
            cleaned_tool_calls = []
            for tool_call in response.tool_calls:
                # 只保留有效的参数
                valid_args = {}
                # 根据工具名称处理参数
                if tool_call["name"] == "rag_knowledge_search":
                    # 仅保留query参数
                    if "query" in tool_call["args"]:
                        valid_args["query"] = tool_call["args"]["query"]
                    else:
                        # 如果没有query参数，使用第一个参数或整个内容作为查询
                        first_arg = next(iter(tool_call["args"].values()), "未知查询")
                        valid_args["query"] = str(first_arg)
                        logger.warning(f"rag_knowledge_search缺少query参数，使用备用参数: {first_arg}")
                elif tool_call["name"] == "web_search_tool":
                    # 仅保留query参数
                    if "query" in tool_call["args"]:
                        valid_args["query"] = tool_call["args"]["query"]
                    else:
                        first_arg = next(iter(tool_call["args"].values()), "未知查询")
                        valid_args["query"] = str(first_arg)
                        logger.warning(f"web_search_tool缺少query参数，使用备用参数: {first_arg}")
                # 添加用户文档工具参数处理
                elif tool_call["name"].startswith("search_"):
                    # 仅保留query参数
                    if "query" in tool_call["args"]:
                        valid_args["query"] = tool_call["args"]["query"]
                    else:
                        first_arg = next(iter(tool_call["args"].values()), "未知查询")
                        valid_args["query"] = str(first_arg)
                        logger.warning(f"{tool_call['name']}缺少query参数，使用备用参数: {first_arg}")

                # 创建清理后的工具调用
                cleaned_tool_call = {
                    "name": tool_call["name"],
                    "args": valid_args,
                    "id": tool_call["id"]
                }
                cleaned_tool_calls.append(cleaned_tool_call)
            # 替换原始的tool_calls
            response.tool_calls = cleaned_tool_calls
            logger.info(f"已清理工具调用参数，移除无效参数")

        # 计算工具调用增量
        tool_call_increment = 1 if (hasattr(response, "tool_calls") and response.tool_calls) else 0

        return {
            "messages": [response],
            "tool_call_count": tool_call_increment,
            "knowledge_base_name": knowledge_base_name,  # 确保传递知识库名称
            "user_document_tools": user_document_tools_list  # 确保传递用户文档工具列表
        }


# 5. 修改条件函数
def should_continue(state: AgentState):
    """决定是否需要调用工具或结束"""
    messages = state["messages"]
    last_message = messages[-1]
    # 检查工具调用次数 - 超过3次强制结束
    if state.get("tool_call_count", 0) >= 5:
        return END
    # 如果有工具调用，则继续
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # 否则结束
    return END


# 6. 全局变量存储编译后的图
graph = None
checkpointer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph, checkpointer
    pool = None
    try:
        # 使用 AsyncConnectionPool 替代 SQLAlchemy 引擎
        DB_URI = "postgresql://postgres:postgres@localhost:5432/langgraph_db?sslmode=disable"
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        # 创建异步连接池
        pool = AsyncConnectionPool(
            conninfo=DB_URI,
            max_size=20,
            kwargs=connection_kwargs
        )
        await pool.open()  # 替代pool.wait()
        # 正确初始化 AsyncPostgresSaver (使用工厂方法)
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        logger.info("✅ 数据库检查点存储初始化成功")

        # 只创建web_search_tool，rag_tool将在call_model中动态创建
        from searxng_server import create_search_tool
        web_search_tool = create_search_tool()
        logger.info("✅ 工具加载成功: web_search")

        builder = StateGraph(AgentState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", tool_node)  # 只绑定web_search_tool
        builder.add_edge(START, "call_model")
        builder.add_conditional_edges(
            "call_model",
            should_continue,
            {
                "tools": "tools",
                END: END
            }
        )
        builder.add_edge("tools", "call_model")
        # 编译图
        graph = builder.compile(checkpointer=checkpointer)
        logger.info("✅ Agent图编译成功")
        yield  # 应用运行中
    except Exception as e:
        logger.error(f"初始化失败: {str(e)}", exc_info=True)
        raise
    finally:
        if pool:
            await pool.close()
            logger.info("✅ 数据库连接池已关闭")


# 7. 创建FastAPI应用
app = FastAPI(
    title="多轮对话Agent API",
    description="基于LangGraph的异步多轮对话Agent服务",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求/响应模型
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: Optional[bool] = False  # 是否启用流式响应
    knowledge_base_name: Optional[str] = "test"  # 新增参数，默认为"test"
    url: Optional[str] = None
    enable_web_search: Optional[bool] = True
    message_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    conversation_history: List[Dict[str, str]]
    tool_calls: Optional[List[Dict]] = None


class ErrorResponse(BaseModel):
    detail: str


def tool_node(state: AgentState):
    """自定义工具节点，能根据知识库名称动态获取工具"""
    messages = state["messages"]
    last_message = messages[-1]
    knowledge_base_name = state.get("knowledge_base_name", "test")

    # 动态获取当前知识库的RAG工具
    rag_tool = get_rag_tool(knowledge_base_name)


    # 创建工具映射
    tools = {
        "rag_knowledge_search": rag_tool
    }
    # 仅当启用时才添加web搜索工具
    if state.get("web_search_enabled", True):
        tools["web_search_tool"] = web_search_tool

    # 添加用户文档工具
    user_document_tools_list = state.get("user_document_tools", [])
    for tool_name in user_document_tools_list:
        tool_info = get_user_document_tool(tool_name)
        if tool_info and "tool" in tool_info:
            tools[tool_info["tool"].name] = tool_info["tool"]

    # 执行所有工具调用
    outputs = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        if tool_name in tools:
            tool = tools[tool_name]
            try:
                # 调用工具
                response = tool.invoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=str(response),
                        name=tool_name,
                        tool_call_id=tool_call["id"]
                    )
                )
            except Exception as e:
                outputs.append(
                    ToolMessage(
                        content=f"工具调用错误: {str(e)}",
                        name=tool_name,
                        status="error",
                        tool_call_id=tool_call["id"]
                    )
                )
        else:
            outputs.append(
                ToolMessage(
                    content=f"Error: {tool_name} is not a valid tool, try one of [{', '.join(tools.keys())}]",
                    name=tool_name,
                    status="error"
                )
            )
    return {"messages": outputs}
# 8. 核心API端点
@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "无效的会话ID"},
        500: {"model": ErrorResponse, "description": "内部服务错误"}
    }
)
async def chat_endpoint(request: ChatRequest):
    """
    处理聊天请求
    - 新会话: 不提供session_id
    - 续会话: 提供有效的session_id
    - url: 可选，用户上传的文档URL
    """
    global graph
    if not graph:
        raise HTTPException(
            status_code=503,
            detail="服务尚未初始化完成"
        )
    # 生成/验证会话ID
    session_id = request.session_id or f"session_{uuid.uuid4()}"
    config = {"configurable": {"thread_id": session_id}}
    try:
        # 获取当前会话状态
        state = await graph.aget_state(config)
        # 构建新状态
        if state is None or not isinstance(state.values, dict) or "messages" not in state.values:
            # 新会话（包括状态不完整的情况）
            user_document_tools_list = []

            # =============== 新增：处理文档URL ===============
            if request.url:
                try:
                    logger.info(f"处理用户上传的文档URL: {request.url}")
                    # 使用session_id作为document_id
                    tool_name = register_user_document_tool(
                        url=request.url,
                        document_id=session_id,
                        document_name="用户上传的基因检测报告"
                    )
                    user_document_tools_list.append(tool_name)
                    logger.info(f"成功注册用户文档工具: {tool_name}")
                except Exception as e:
                    logger.error(f"文档处理失败: {str(e)}")
                    # 即使文档处理失败，也要继续对话
            # =============== 新增结束 ===============

            initial_state = {
                "messages": [HumanMessage(content=request.message)],
                "knowledge_base_name": request.knowledge_base_name,
                "tool_call_count": 0,
                "user_document_tools": user_document_tools_list,
                "web_search_enabled": request.enable_web_search  # 保存web搜索开关状态
            }
        else:
            # 续会话 - 复制现有状态并添加新消息
            # 保留之前的知识库名称，即使请求中提供了新值（避免中途切换知识库导致混淆）
            knowledge_base_name = state.values.get("knowledge_base_name", request.knowledge_base_name)

            # =============== 新增：处理文档URL（如果是新上传的文档） ===============
            user_document_tools_list = state.values.get("user_document_tools", [])

            # 如果有新的URL且还没有注册过对应的工具
            if request.url and not any(
                    tool_name.startswith("search_" + session_id) for tool_name in user_document_tools_list):
                try:
                    logger.info(f"处理用户上传的文档URL: {request.url}")
                    # 使用session_id作为document_id
                    tool_name = register_user_document_tool(
                        url=request.url,
                        document_id=session_id,
                        document_name="用户上传的基因检测报告"
                    )
                    user_document_tools_list.append(tool_name)
                    logger.info(f"成功注册用户文档工具: {tool_name}")
                except Exception as e:
                    logger.error(f"文档处理失败: {str(e)}")
            # =============== 新增结束 ===============
            # 保留之前的web搜索设置，除非请求中提供了新值
            web_search_enabled = state.values.get("web_search_enabled", request.enable_web_search)
            initial_state = {
                "messages": state.values["messages"] + [HumanMessage(content=request.message)],
                "knowledge_base_name": knowledge_base_name,
                "tool_call_count": state.values.get("tool_call_count", 0),
                "user_document_tools": user_document_tools_list,
                "web_search_enabled": web_search_enabled
            }
        # 执行对话流
        final_state = None
        async for step in graph.astream(initial_state, config=config, stream_mode="values"):
            final_state = step
        if not final_state:
            raise HTTPException(
                status_code=500,
                detail="对话流程未产生有效响应"
            )
        # 提取最新回复
        last_msg = final_state["messages"][-1]
        if not isinstance(last_msg, AIMessage):
            raise HTTPException(
                status_code=500,
                detail="无效的模型响应类型"
            )
        # 收集工具调用信息（用于调试）
        tool_calls = []
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            for tool_call in last_msg.tool_calls:
                tool_calls.append({
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                    "id": tool_call["id"]
                })
        # 构建历史记录
        history = []
        for msg in final_state["messages"]:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                history.append({"role": "system", "content": "系统提示"})
            elif isinstance(msg, ToolMessage):
                history.append({"role": "tool", "content": msg.content})
        return ChatResponse(
            response=last_msg.content,
            session_id=session_id,
            conversation_history=history,
            tool_calls=tool_calls if tool_calls else None
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 处理请求失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"请求处理失败: {str(e)}"
        )


# 9. 流式响应API（可选）
@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    流式响应聊天请求（固定分块大小）
    - 使用固定大小的文本块进行流式传输
    - 符合标准SSE流式输出格式
    """
    global graph
    if not graph:
        raise HTTPException(
            status_code=503,
            detail="服务尚未初始化完成"
        )
    # 生成/验证会话ID
    session_id = request.session_id or f"session_{uuid.uuid4()}"
    message_id = request.message_id
    config = {"configurable": {"thread_id": session_id}}

    async def event_generator():
        try:
            # 获取当前会话状态和构建初始状态（与之前相同）
            state = await graph.aget_state(config)
            if state is None or not isinstance(state.values, dict) or "messages" not in state.values:
                # 新会话
                user_document_tools_list = []
                if request.url:
                    try:
                        logger.info(f"处理用户上传的文档URL: {request.url}")
                        document_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
                        tool_name = register_user_document_tool(
                            url=request.url,
                            document_id=document_id,
                            document_name="用户上传的基因检测报告"
                        )
                        user_document_tools_list.append(tool_name)
                        logger.info(f"成功注册用户文档工具: {tool_name}")
                    except Exception as e:
                        logger.error(f"文档处理失败: {str(e)}")
                initial_state = {
                    "messages": [HumanMessage(content=request.message)],
                    "knowledge_base_name": request.knowledge_base_name,
                    "tool_call_count": 0,
                    "user_document_tools": user_document_tools_list,
                    "web_search_enabled": request.enable_web_search
                }
            else:
                # 续会话
                knowledge_base_name = state.values.get("knowledge_base_name", request.knowledge_base_name)
                user_document_tools_list = state.values.get("user_document_tools", [])
                web_search_enabled = state.values.get("web_search_enabled", request.enable_web_search)
                if request.url and not any(
                        tool_name.startswith("search_" + session_id) for tool_name in user_document_tools_list):
                    try:
                        logger.info(f"处理用户上传的文档URL: {request.url}")
                        document_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
                        tool_name = register_user_document_tool(
                            url=request.url,
                            document_id=document_id,
                            document_name="用户上传的基因检测报告"
                        )
                        user_document_tools_list.append(tool_name)
                        logger.info(f"成功注册用户文档工具: {tool_name}")
                    except Exception as e:
                        logger.error(f"文档处理失败: {str(e)}")
                initial_state = {
                    "messages": state.values["messages"] + [HumanMessage(content=request.message)],
                    "knowledge_base_name": knowledge_base_name,
                    "tool_call_count": state.values.get("tool_call_count", 0),
                    "user_document_tools": user_document_tools_list,
                    "web_search_enabled": web_search_enabled
                }

            # 固定分块大小设置为64个字符
            CHUNK_SIZE = 10
            full_text = ""

            async for step in graph.astream_log(initial_state, config=config):
                if isinstance(step, dict) and "ops" in step:
                    ops = step["ops"]
                elif hasattr(step, "ops"):
                    ops = step.ops
                else:
                    continue

                for op in ops:
                    if isinstance(op, dict):
                        path = op.get("path", "")
                        value = op.get("value")
                    elif hasattr(op, "path") and hasattr(op, "value"):
                        path = op.path
                        value = op.value
                    else:
                        continue

                    if path.startswith("/logs/call_model/") and value is not None:
                        if isinstance(value, dict) and "messages" in value:
                            for msg in value["messages"]:
                                if isinstance(msg, AIMessage) and hasattr(msg, "content"):
                                    full_text += msg.content

                                    # 固定分块处理
                                    while len(full_text) >= CHUNK_SIZE:
                                        chunk = full_text[:CHUNK_SIZE]
                                        full_text = full_text[CHUNK_SIZE:]
                                        logger.info(f"发送流式内容: {full_text}")
                                        # 构建符合规范的SSE数据块
                                        data = {
                                            "text": chunk,
                                            "finish_reason": None,
                                            "session_id": session_id,
                                            "message_id": message_id
                                        }
                                        yield f"data: {json.dumps(data)}\n\n"
                                        await asyncio.sleep(0.01)

            # 发送剩余内容（如果有）
            if full_text:
                data = {
                    "text": full_text,
                    "finish_reason": None,
                    "session_id": session_id,
                    "message_id": message_id
                }
                yield f"data: {json.dumps(data)}\n\n"

            # 发送结束标记
            end_data = {
                "text": "",
                "finish_reason": "stop",
                "session_id": session_id,
                "message_id": message_id
            }
            yield f"data: {json.dumps(end_data)}\n\n"

        except Exception as e:
            logger.error(f"❌ 流式响应失败: {str(e)}", exc_info=True)
            error_data = {
                "text": "",
                "finish_reason": "error",
                "session_id": session_id,
                "message_id": message_id,
                "error": str(e)
            }
            yield f"event: error\n"
            yield f"data: {json.dumps(error_data)}\n\n"

    # 确保导入json模块
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


# 10. 会话管理API
@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """获取特定会话的完整历史"""
    global graph
    if not graph:
        raise HTTPException(
            status_code=503,
            detail="服务尚未初始化完成"
        )
    config = {"configurable": {"thread_id": session_id}}
    try:
        state = await graph.aget_state(config)
        if not state or not isinstance(state.values, dict) or "messages" not in state.values:
            raise HTTPException(
                status_code=404,
                detail="会话不存在"
            )
        history = []
        for msg in state.values["messages"]:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                history.append({"role": "system", "content": "系统提示"})
            elif isinstance(msg, ToolMessage):
                history.append({"role": "tool", "content": msg.content})
        return {
            "session_id": session_id,
            "conversation_history": history,
            "last_updated": state.config["configurable"].get("checkpoint_id")
        }
    except Exception as e:
        logger.error(f"❌ 获取会话失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取会话失败: {str(e)}"
        )

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除特定会话"""
    global checkpointer
    if not checkpointer:
        raise HTTPException(
            status_code=503,
            detail="服务尚未初始化完成"
        )
    try:
        # 删除会话
        await checkpointer.aput(
            {"configurable": {"thread_id": session_id}},
            None,  # 传递None表示删除
            None  # 传递None表示删除
        )
        return {"status": "success", "message": f"会话 {session_id} 已删除"}
    except Exception as e:
        logger.error(f"❌ 删除会话失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"删除会话失败: {str(e)}"
        )

# 11. 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model": "qwen3:4b",
        "tools": ["rag_knowledge_search", "web_search_tool"],
        "database": "connected" if checkpointer else "disconnected"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("启动API服务...")
    uvicorn.run(
        "ollama:app",  # 注意这里：文件名:变量名
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        log_level="info"
    )