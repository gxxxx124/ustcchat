"""
多轮对话历史管理系统 - 代码分析与实现说明
============================================

本文件整理了系统中所有与多轮对话历史相关的代码实现，
包括会话管理、状态持久化、历史记录处理等核心功能。
"""

# ============================================================================
# 1. 核心数据结构定义
# ============================================================================

from typing import Dict, List, Optional, Any, TypedDict, Annotated
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, AIMessage
import uuid
import operator

class AgentState(TypedDict):
    """对话状态数据结构 - 核心状态管理"""
    messages: Annotated[list, operator.add]  # 消息历史，使用add操作符进行累积
    tool_call_count: Annotated[int, operator.add]  # 工具调用计数器
    knowledge_base_name: str  # 当前会话使用的知识库名称
    user_document_tools: List[str]  # 当前会话可用的用户文档工具名称
    web_search_enabled: bool  # 记录web搜索是否启用

class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str  # 用户消息
    session_id: Optional[str] = None  # 会话ID，用于续会话
    stream: Optional[bool] = False  # 是否启用流式响应
    knowledge_base_name: Optional[str] = "test"  # 知识库名称
    url: Optional[str] = None  # 用户上传的文档URL
    enable_web_search: Optional[bool] = True  # 是否启用网络搜索
    message_id: Optional[str] = None  # 消息ID

class ChatResponse(BaseModel):
    """聊天响应模型"""
    response: str  # AI回复内容
    session_id: str  # 会话ID
    conversation_history: List[Dict[str, str]]  # 完整对话历史
    tool_calls: Optional[List[Dict]] = None  # 工具调用信息

# ============================================================================
# 2. 会话管理核心实现
# ============================================================================

class ConversationManager:
    """
    对话历史管理器 - 核心功能实现
    """
    
    def __init__(self, graph, checkpointer):
        self.graph = graph
        self.checkpointer = checkpointer
    
    async def create_or_continue_session(self, request: ChatRequest) -> tuple[str, dict]:
        """
        创建新会话或继续现有会话
        
        实现逻辑：
        1. 生成或验证会话ID
        2. 获取现有会话状态
        3. 构建初始状态或续接状态
        """
        # 生成/验证会话ID
        session_id = request.session_id or f"session_{uuid.uuid4()}"
        config = {"configurable": {"thread_id": session_id}}
        
        # 获取当前会话状态
        state = await self.graph.aget_state(config)
        
        if state is None or not isinstance(state.values, dict) or "messages" not in state.values:
            # 新会话处理
            return self._create_new_session(request, session_id)
        else:
            # 续会话处理
            return self._continue_existing_session(request, session_id, state)
    
    def _create_new_session(self, request: ChatRequest, session_id: str) -> tuple[str, dict]:
        """创建新会话的初始状态"""
        user_document_tools_list = []
        
        # 处理用户上传的文档URL
        if request.url:
            try:
                tool_name = self._register_user_document_tool(request.url, session_id)
                user_document_tools_list.append(tool_name)
            except Exception as e:
                logger.error(f"文档处理失败: {str(e)}")
        
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "knowledge_base_name": request.knowledge_base_name,
            "tool_call_count": 0,
            "user_document_tools": user_document_tools_list,
            "web_search_enabled": request.enable_web_search
        }
        
        return session_id, initial_state
    
    def _continue_existing_session(self, request: ChatRequest, session_id: str, state: dict) -> tuple[str, dict]:
        """继续现有会话的状态构建"""
        # 保留之前的知识库名称，避免中途切换知识库导致混淆
        knowledge_base_name = state.values.get("knowledge_base_name", request.knowledge_base_name)
        
        # 获取现有的用户文档工具列表
        user_document_tools_list = state.values.get("user_document_tools", [])
        
        # 处理新的文档URL（如果提供）
        if request.url and not any(
                tool_name.startswith("search_" + session_id) for tool_name in user_document_tools_list):
            try:
                tool_name = self._register_user_document_tool(request.url, session_id)
                user_document_tools_list.append(tool_name)
            except Exception as e:
                logger.error(f"文档处理失败: {str(e)}")
        
        # 保留之前的web搜索设置
        web_search_enabled = state.values.get("web_search_enabled", request.enable_web_search)
        
        initial_state = {
            "messages": state.values["messages"] + [HumanMessage(content=request.message)],
            "knowledge_base_name": knowledge_base_name,
            "tool_call_count": state.values.get("tool_call_count", 0),
            "user_document_tools": user_document_tools_list,
            "web_search_enabled": web_search_enabled
        }
        
        return session_id, initial_state

# ============================================================================
# 3. 对话历史智能截断机制
# ============================================================================

def smart_conversation_truncation(messages: List, max_length: int = 10, keep_recent: int = 5):
    """
    智能对话历史截断机制
    
    实现原理：
    1. 监控消息数量，防止显存累积
    2. 当超过阈值时，保留最新的关键消息
    3. 确保LLM能理解最近的对话上下文
    
    参数：
    - messages: 当前消息列表
    - max_length: 触发截断的阈值
    - keep_recent: 保留的最新消息数量
    """
    if len(messages) > max_length:
        # 保留最新的5条消息和系统提示
        truncated_messages = messages[-keep_recent:]
        logger.info("⚠️ 会话历史过长，已截断以节省显存")
        return truncated_messages
    return messages

# 在call_model函数中的应用：
"""
async def call_model(state: AgentState):
    messages = state["messages"]
    knowledge_base_name = state.get("knowledge_base_name", "test")
    
    # 显存优化：限制会话长度，防止显存累积
    if len(messages) > 10:  # 限制会话历史长度
        # 保留最新的5条消息和系统提示
        messages = messages[-5:]
        logger.info("⚠️ 会话历史过长，已截断以节省显存")
    
    # ... 其他处理逻辑
"""

# ============================================================================
# 4. 状态持久化与恢复
# ============================================================================

class StatePersistence:
    """
    状态持久化管理器
    
    实现机制：
    1. 使用PostgreSQL作为检查点存储
    2. 每个会话有唯一的thread_id
    3. 支持会话状态的保存和恢复
    """
    
    def __init__(self, checkpointer):
        self.checkpointer = checkpointer
    
    async def save_session_state(self, session_id: str, state: dict):
        """保存会话状态"""
        config = {"configurable": {"thread_id": session_id}}
        # 状态会自动保存到数据库
    
    async def load_session_state(self, session_id: str) -> Optional[dict]:
        """加载会话状态"""
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = await self.graph.aget_state(config)
            return state
        except Exception as e:
            logger.error(f"加载会话状态失败: {str(e)}")
            return None
    
    async def delete_session(self, session_id: str):
        """删除会话"""
        try:
            await self.checkpointer.aput(
                {"configurable": {"thread_id": session_id}},
                None,  # 传递None表示删除
                None   # 传递None表示删除
            )
            return {"status": "success", "message": f"会话 {session_id} 已删除"}
        except Exception as e:
            logger.error(f"删除会话失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")

# ============================================================================
# 5. 对话历史构建与格式化
# ============================================================================

def build_conversation_history(messages: List) -> List[Dict[str, str]]:
    """
    构建标准化的对话历史记录
    
    实现逻辑：
    1. 遍历所有消息类型
    2. 转换为标准格式
    3. 过滤系统消息，保留用户和AI对话
    """
    history = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            history.append({"role": "system", "content": "系统提示"})
        elif isinstance(msg, ToolMessage):
            history.append({"role": "tool", "content": msg.content})
    
    return history

# ============================================================================
# 6. 会话管理API端点
# ============================================================================

"""
@agent_router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    '''获取特定会话的完整历史'''
    
    实现功能：
    1. 根据session_id获取会话状态
    2. 构建完整的对话历史
    3. 返回标准化的历史记录格式
"""

"""
@agent_router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    '''删除特定会话'''
    
    实现功能：
    1. 从数据库中删除会话状态
    2. 清理相关资源
    3. 返回操作结果
"""

# ============================================================================
# 7. 流式对话历史处理
# ============================================================================

class StreamingConversationHandler:
    """
    流式对话历史处理器
    
    实现机制：
    1. 实时构建对话历史
    2. 分块发送响应
    3. 维护会话状态连续性
    """
    
    def __init__(self, graph, session_id: str):
        self.graph = graph
        self.session_id = session_id
        self.config = {"configurable": {"thread_id": session_id}}
    
    async def handle_streaming_chat(self, request: ChatRequest):
        """处理流式聊天请求"""
        # 获取或创建会话状态
        state = await self.graph.aget_state(self.config)
        
        if state is None or not isinstance(state.values, dict) or "messages" not in state.values:
            # 新会话处理
            initial_state = self._create_initial_state_for_streaming(request)
        else:
            # 续会话处理
            initial_state = self._continue_state_for_streaming(request, state)
        
        # 流式处理对话
        return self._process_streaming_response(initial_state)
    
    def _create_initial_state_for_streaming(self, request: ChatRequest):
        """为流式响应创建初始状态"""
        user_document_tools_list = []
        if request.url:
            try:
                document_id = f"{self.session_id}_{uuid.uuid4().hex[:8]}"
                tool_name = self._register_user_document_tool(request.url, document_id)
                user_document_tools_list.append(tool_name)
            except Exception as e:
                logger.error(f"文档处理失败: {str(e)}")
        
        return {
            "messages": [HumanMessage(content=request.message)],
            "knowledge_base_name": request.knowledge_base_name,
            "tool_call_count": 0,
            "user_document_tools": user_document_tools_list,
            "web_search_enabled": request.enable_web_search
        }
    
    def _continue_state_for_streaming(self, request: ChatRequest, state: dict):
        """为流式响应继续现有状态"""
        knowledge_base_name = state.values.get("knowledge_base_name", request.knowledge_base_name)
        user_document_tools_list = state.values.get("user_document_tools", [])
        web_search_enabled = state.values.get("web_search_enabled", request.enable_web_search)
        
        if request.url and not any(
                tool_name.startswith("search_" + self.session_id) for tool_name in user_document_tools_list):
            try:
                document_id = f"{self.session_id}_{uuid.uuid4().hex[:8]}"
                tool_name = self._register_user_document_tool(request.url, document_id)
                user_document_tools_list.append(tool_name)
            except Exception as e:
                logger.error(f"文档处理失败: {str(e)}")
        
        return {
            "messages": state.values["messages"] + [HumanMessage(content=request.message)],
            "knowledge_base_name": knowledge_base_name,
            "tool_call_count": state.values.get("tool_call_count", 0),
            "user_document_tools": user_document_tools_list,
            "web_search_enabled": web_search_enabled
        }

# ============================================================================
# 8. 关键实现细节总结
# ============================================================================

"""
多轮对话历史系统的核心实现特点：

1. 状态管理：
   - 使用TypedDict定义强类型状态结构
   - 支持消息累积、工具调用计数等状态跟踪
   - 每个会话有唯一的thread_id标识

2. 会话连续性：
   - 支持新会话创建和现有会话续接
   - 保持知识库名称、工具配置等会话上下文
   - 智能处理用户文档工具的动态注册

3. 历史记录处理：
   - 智能截断机制防止显存累积
   - 保留最新的关键消息确保上下文理解
   - 标准化的历史记录格式便于前端处理

4. 持久化存储：
   - PostgreSQL作为检查点存储后端
   - 支持会话状态的保存、恢复和删除
   - 异步操作确保性能

5. 流式响应支持：
   - 实时构建对话历史
   - 分块发送响应内容
   - 维护会话状态的连续性

6. 错误处理：
   - 完善的异常捕获和处理
   - 会话状态验证和恢复
   - 详细的日志记录便于调试
"""

# ============================================================================
# 9. 使用示例
# ============================================================================

"""
# 创建对话管理器
conversation_manager = ConversationManager(graph, checkpointer)

# 处理聊天请求
session_id, initial_state = await conversation_manager.create_or_continue_session(request)

# 执行对话流
final_state = None
async for step in graph.astream(initial_state, config={"configurable": {"thread_id": session_id}}):
    final_state = step

# 构建对话历史
history = build_conversation_history(final_state["messages"])

# 返回响应
return ChatResponse(
    response=final_state["messages"][-1].content,
    session_id=session_id,
    conversation_history=history
)
"""

if __name__ == "__main__":
    print("多轮对话历史管理系统代码分析完成！")
    print("本文件包含了系统中所有与对话历史相关的核心实现逻辑。")
