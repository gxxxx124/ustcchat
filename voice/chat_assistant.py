"""
智能对话助手
可以在另一台服务器上运行，连接到远程数据库存储对话历史
使用相同的 DeepSeek API 接入方式
"""
import os
import sys
import asyncio
import logging
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

# 添加路径以便导入模块
parent_dir = os.path.join(os.path.dirname(__file__), '..')
ustc_dir = os.path.join(parent_dir, 'ustc')
sys.path.insert(0, parent_dir)
sys.path.insert(0, ustc_dir)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from psycopg_pool import AsyncConnectionPool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.tools import Tool
from nsrl_deepseek_client import NSRLDeepSeekChat
import requests

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="智能对话助手",
    description="连接到远程数据库的智能对话助手",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
chat_history_pool: Optional[AsyncConnectionPool] = None
deepseek_model: Optional[NSRLDeepSeekChat] = None
mcp_knowledge_server_url: str = ""


# ==================== 数据模型 ====================

class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str = Field(..., description="消息角色: user 或 assistant")
    content: str = Field(..., description="消息内容")


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="用户消息")
    session_id: Optional[str] = Field(None, description="会话ID，不提供则创建新会话")
    user_id: int = Field(default=1, description="用户ID")
    system_prompt: Optional[str] = Field(None, description="系统提示词")


class ChatResponse(BaseModel):
    """聊天响应模型"""
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="助手回复")
    conversation_history: List[ChatMessage] = Field(default_factory=list, description="对话历史")


class SessionListResponse(BaseModel):
    """会话列表响应"""
    sessions: List[Dict[str, Any]] = Field(default_factory=list, description="会话列表")


# ==================== 初始化函数 ====================

def init_deepseek_model():
    """初始化 DeepSeek 模型"""
    global deepseek_model
    
    if deepseek_model is not None:
        return deepseek_model
    
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key or api_key.strip() == "":
        raise ValueError("❌ DEEPSEEK_API_KEY 未配置！请在 .env 文件中设置 DEEPSEEK_API_KEY 环境变量。")
    
    api_base = os.getenv("DEEPSEEK_API_BASE", "http://scc.ustc.edu.cn/portal/api/ask")
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-v3")
    
    deepseek_model = NSRLDeepSeekChat(
        api_key=api_key,
        api_base=api_base,
        model=model_name,
        max_tokens=10000,
        temperature=0.1,
        request_timeout=120.0,
        max_retries=5,
    )
    
    logger.info(f"🚀 DeepSeek 模型初始化成功: {model_name} (端点: {api_base})")
    return deepseek_model


async def init_database():
    """初始化数据库连接池"""
    global chat_history_pool
    
    if chat_history_pool is not None:
        return chat_history_pool
    
    # 从环境变量读取数据库连接字符串
    # 格式: postgresql://user:password@host:port/database?sslmode=disable
    CHAT_HISTORY_DB_URI = os.getenv(
        "CHAT_HISTORY_DB_URI",
        "postgresql://chat_history_user:chat_history_pass@localhost:5432/chat_history_db?sslmode=disable"
    )
    
    logger.info(f"初始化对话历史数据库连接池...")
    logger.info(f"数据库地址: {CHAT_HISTORY_DB_URI.split('@')[1] if '@' in CHAT_HISTORY_DB_URI else 'localhost'}")
    
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }
    
    chat_history_pool = AsyncConnectionPool(
        conninfo=CHAT_HISTORY_DB_URI,
        max_size=10,
        kwargs=connection_kwargs
    )
    
    await chat_history_pool.open()
    
    # 确保表存在
    async with chat_history_pool.connection() as conn:
        async with conn.cursor() as cur:
            # 创建 conversations 表
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id VARCHAR(255) PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    title VARCHAR(500),
                    message_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建 messages 表
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    conversation_id VARCHAR(255) NOT NULL,
                    role VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                )
            """)
            
            # 创建索引
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
                ON messages(conversation_id)
            """)
            
            await cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_user_id 
                ON conversations(user_id)
            """)
            
            await conn.commit()
    
    logger.info("✅ 对话历史数据库连接池初始化成功")
    return chat_history_pool


async def get_conversation_history(session_id: str) -> List[Dict[str, str]]:
    """获取对话历史"""
    if chat_history_pool is None:
        return []
    
    try:
        async with chat_history_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT role, content
                    FROM messages 
                    WHERE conversation_id = %s 
                    ORDER BY created_at ASC
                """, (session_id,))
                
                rows = await cur.fetchall()
                history = []
                for row in rows:
                    role, content = row
                    history.append({"role": role, "content": content})
                
                return history
    except Exception as e:
        logger.error(f"获取对话历史失败: {str(e)}")
        return []


async def save_message(session_id: str, role: str, content: str, user_id: int = 1, title: Optional[str] = None):
    """保存消息到数据库"""
    if chat_history_pool is None:
        return
    
    try:
        async with chat_history_pool.connection() as conn:
            async with conn.cursor() as cur:
                # 插入或更新对话记录
                await cur.execute("""
                    INSERT INTO conversations (conversation_id, user_id, title, message_count, updated_at)
                    VALUES (%s, %s, %s, 1, CURRENT_TIMESTAMP)
                    ON CONFLICT (conversation_id) 
                    DO UPDATE SET 
                        title = COALESCE(EXCLUDED.title, conversations.title),
                        message_count = conversations.message_count + 1,
                        updated_at = CURRENT_TIMESTAMP
                """, (session_id, user_id, title))
                
                # 插入消息
                await cur.execute("""
                    INSERT INTO messages (conversation_id, role, content)
                    VALUES (%s, %s, %s)
                """, (session_id, role, content))
                
                await conn.commit()
                logger.debug(f"消息已保存: {session_id}, {role}")
    except Exception as e:
        logger.error(f"保存消息失败: {str(e)}")


# ==================== MCP 知识库工具 ====================

def create_knowledge_search_tool(mcp_server_url: str):
    """创建知识库搜索工具（通过 MCP 服务器）"""
    
    def knowledge_search(query: str) -> str:
        """在知识库中搜索相关信息"""
        try:
            # 调用 MCP 服务器的 RESTful API
            # 使用默认参数
            collection_name = "nsrl_tech_docs"
            k = 15
            
            response = requests.post(
                f"{mcp_server_url}/api/search",
                json={
                    "query": query,
                    "collection_name": collection_name,
                    "k": k,
                    "title_weight": 0.6,
                    "content_weight": 0.4
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code != 200:
                return f"知识库搜索失败: HTTP {response.status_code}"
            
            result = response.json()
            if not result.get("success"):
                return "知识库搜索失败: 未找到相关信息"
            
            # 格式化结果
            results = result.get("results", [])
            if not results:
                return "未在知识库中找到相关信息。"
            
            formatted_results = []
            for item in results:
                metadata = item.get("metadata", {})
                content = item.get("content", "")
                score = item.get("score", 0)
                
                if metadata.get("is_qa_pair"):
                    formatted_results.append(
                        f"【QA对 - 相似度: {score:.4f}】\n"
                        f"来源: {metadata.get('source', '未知')}\n"
                        f"内容: {content}\n"
                        "----------------------------------------"
                    )
                else:
                    formatted_results.append(
                        f"【文档片段 - 相似度: {score:.4f}】\n"
                        f"标题: {metadata.get('title', '无标题')}\n"
                        f"内容: {content}\n"
                        f"来源: {metadata.get('source', '未知')}\n"
                        "----------------------------------------"
                    )
            
            return f"📚 在知识库中找到 {len(results)} 个相关结果：\n\n" + "\n".join(formatted_results)
        
        except requests.exceptions.RequestException as e:
            logger.error(f"MCP 知识库搜索请求失败: {str(e)}")
            return f"知识库搜索失败: 无法连接到MCP服务器 ({str(e)})"
        except Exception as e:
            logger.error(f"知识库搜索失败: {str(e)}", exc_info=True)
            return f"知识库搜索失败: {str(e)}"
    
    # 定义工具输入模式
    from pydantic import BaseModel
    
    class KnowledgeSearchInput(BaseModel):
        query: str = Field(..., description="搜索查询语句")
    
    return Tool.from_function(
        name="knowledge_base_search",
        description="在知识库中搜索相关信息。当用户询问关于文档、配置、流程、技术问题等问题时，使用此工具搜索知识库中的相关内容。输入应该是用户的查询问题。",
        func=knowledge_search,
        args_schema=KnowledgeSearchInput,
        return_direct=False
    )


# ==================== API 端点 ====================

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global mcp_knowledge_server_url
    
    logger.info("正在初始化智能对话助手...")
    
    # 初始化数据库
    await init_database()
    
    # 初始化 DeepSeek 模型
    init_deepseek_model()
    
    # 配置 MCP 知识库服务器地址
    mcp_knowledge_server_url = os.getenv(
        "MCP_KNOWLEDGE_SERVER_URL",
        "http://114.214.168.134:8001"
    )
    logger.info(f"MCP 知识库服务器: {mcp_knowledge_server_url}")
    
    logger.info("✅ 智能对话助手初始化完成")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global chat_history_pool
    if chat_history_pool:
        await chat_history_pool.close()
        logger.info("数据库连接池已关闭")


@app.get("/")
async def root():
    """根端点"""
    return {
        "service": "智能对话助手",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/api/chat",
            "sessions": "/api/sessions",
            "health": "/health"
        },
        "features": {
            "tool_calling": True,
            "knowledge_base_search": True,
            "mcp_integration": True
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        if chat_history_pool is None:
            return {"status": "initializing", "database": "not_connected"}
        
        # 测试数据库连接
        async with chat_history_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                await cur.fetchone()
        
        # 测试 MCP 知识库服务器连接
        mcp_status = "unknown"
        try:
            mcp_url = os.getenv("MCP_KNOWLEDGE_SERVER_URL", "http://114.214.168.134:8001")
            mcp_response = requests.get(f"{mcp_url}/health", timeout=5)
            if mcp_response.status_code == 200:
                mcp_status = "connected"
            else:
                mcp_status = f"error_{mcp_response.status_code}"
        except:
            mcp_status = "disconnected"
        
        return {
            "status": "healthy",
            "database": "connected",
            "model": "deepseek-v3" if deepseek_model else "not_initialized",
            "mcp_knowledge_server": mcp_status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """智能对话接口（支持工具调用）"""
    try:
        # 生成或使用会话ID
        if request.session_id:
            session_id = request.session_id
        else:
            session_id = f"session_{uuid.uuid4().hex[:16]}"
        
        # 获取对话历史
        history = await get_conversation_history(session_id)
        
        # 构建消息列表
        messages = []
        
        # 添加系统提示词
        if request.system_prompt:
            messages.append(SystemMessage(content=request.system_prompt))
        else:
            messages.append(SystemMessage(content="你是一个有用的AI助手，能够回答用户的问题并提供帮助。你可以使用知识库搜索工具来查找相关信息。"))
        
        # 添加历史消息
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # 添加当前用户消息
        messages.append(HumanMessage(content=request.message))
        
        # 保存用户消息
        await save_message(session_id, "user", request.message, request.user_id)
        
        # 调用 DeepSeek 模型
        if deepseek_model is None:
            raise HTTPException(status_code=500, detail="DeepSeek 模型未初始化")
        
        # 创建知识库搜索工具
        knowledge_tool = create_knowledge_search_tool(mcp_knowledge_server_url)
        
        # 绑定工具到模型
        model_with_tools = deepseek_model.bind_tools([knowledge_tool])
        
        logger.info(f"调用 DeepSeek 模型生成回复（已绑定工具）- Session: {session_id}")
        
        # 第一轮：调用模型（可能返回工具调用）
        response = model_with_tools.invoke(messages)
        
        # 检查是否有工具调用
        tool_calls = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_calls = response.tool_calls
            logger.info(f"检测到 {len(tool_calls)} 个工具调用")
            for i, tc in enumerate(tool_calls):
                logger.info(f"  工具 {i+1}: {tc.get('name', 'unknown')} - {tc.get('args', {})}")
        
        # 如果有工具调用，执行工具
        if tool_calls:
            # 添加模型的响应（包含工具调用）到消息历史
            messages.append(response)
            
            # 执行所有工具调用
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", "")
                
                logger.info(f"执行工具: {tool_name}, 参数: {tool_args}")
                
                if tool_name == "knowledge_base_search":
                    # 调用知识库搜索工具
                    query = tool_args.get("query", "")
                    if not query:
                        # 如果没有query参数，尝试从用户消息中提取
                        query = request.message
                    
                    collection_name = tool_args.get("collection_name", "nsrl_tech_docs")
                    k = tool_args.get("k", 15)
                    
                    # Tool.from_function 创建的工具，直接传入字符串作为query参数
                    # 其他参数通过工具内部处理
                    tool_result = knowledge_tool.invoke(query)
                    
                    # 添加工具结果到消息历史
                    messages.append(ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_id
                    ))
                    logger.info(f"工具执行完成: {tool_name}, 结果长度: {len(tool_result)} 字符")
                else:
                    logger.warning(f"未知工具: {tool_name}")
                    messages.append(ToolMessage(
                        content=f"未知工具: {tool_name}",
                        tool_call_id=tool_id
                    ))
            
            # 第二轮：基于工具结果生成最终回答（不绑定工具，强制生成回答）
            logger.info("基于工具结果生成最终回答...")
            final_response = deepseek_model.invoke(messages)
            assistant_message = final_response.content if hasattr(final_response, 'content') else str(final_response)
        else:
            # 没有工具调用，直接使用响应
            assistant_message = response.content if hasattr(response, 'content') else str(response)
        
        # 保存助手回复
        await save_message(session_id, "assistant", assistant_message, request.user_id)
        
        # 更新对话历史
        history.append({"role": "user", "content": request.message})
        history.append({"role": "assistant", "content": assistant_message})
        
        return ChatResponse(
            session_id=session_id,
            message=assistant_message,
            conversation_history=[ChatMessage(**msg) for msg in history]
        )
    
    except Exception as e:
        logger.error(f"对话处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"对话处理失败: {str(e)}")


@app.get("/api/sessions", response_model=SessionListResponse)
async def list_sessions(user_id: int = 1, limit: int = 20):
    """获取会话列表"""
    if chat_history_pool is None:
        return SessionListResponse(sessions=[])
    
    try:
        async with chat_history_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT conversation_id, title, message_count, updated_at
                    FROM conversations 
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    LIMIT %s
                """, (user_id, limit))
                
                rows = await cur.fetchall()
                sessions = []
                for row in rows:
                    conv_id, title, msg_count, updated_at = row
                    sessions.append({
                        "session_id": conv_id,
                        "title": title or "未命名对话",
                        "message_count": msg_count,
                        "updated_at": str(updated_at)
                    })
                
                return SessionListResponse(sessions=sessions)
    
    except Exception as e:
        logger.error(f"获取会话列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """获取特定会话的完整历史"""
    try:
        history = await get_conversation_history(session_id)
        
        return {
            "session_id": session_id,
            "conversation_history": history
        }
    except Exception as e:
        logger.error(f"获取会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取会话失败: {str(e)}")


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if chat_history_pool is None:
        raise HTTPException(status_code=500, detail="数据库未初始化")
    
    try:
        async with chat_history_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("DELETE FROM conversations WHERE conversation_id = %s", (session_id,))
                await conn.commit()
        
        return {"success": True, "message": f"会话 {session_id} 已删除"}
    except Exception as e:
        logger.error(f"删除会话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")


# ==================== 启动服务器 ====================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("CHAT_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("CHAT_SERVER_PORT", "8002"))
    
    logger.info(f"启动智能对话助手服务器...")
    logger.info(f"监听地址: {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

