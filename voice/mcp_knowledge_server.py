"""
MCP 知识库搜索服务器
提供符合 MCP (Model Context Protocol) 规范的知识库搜索服务
可以通过 HTTP API 供公网服务器调用
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 添加父目录和 ustc 目录到路径，以便导入模块
parent_dir = os.path.join(os.path.dirname(__file__), '..')
ustc_dir = os.path.join(parent_dir, 'ustc')
sys.path.insert(0, parent_dir)
sys.path.insert(0, ustc_dir)

from chunks2embedding import embedding_init
from rag_tool import create_rag_tool

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
    title="MCP Knowledge Base Search Server",
    description="MCP 协议的知识库搜索服务",
    version="1.0.0"
)

# 配置 CORS，允许跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储向量存储和工具
vector_stores = {}
rag_tools = {}


# ==================== MCP 协议模型 ====================

class MCPTool(BaseModel):
    """MCP 工具定义"""
    name: str
    description: str
    inputSchema: Dict[str, Any]


class MCPListToolsResponse(BaseModel):
    """MCP 列出工具响应"""
    tools: List[MCPTool]


class MCPCallToolRequest(BaseModel):
    """MCP 调用工具请求"""
    name: str
    arguments: Dict[str, Any]


class MCPCallToolResponse(BaseModel):
    """MCP 调用工具响应"""
    content: List[Dict[str, Any]]
    isError: bool = False


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str = Field(..., description="搜索查询语句")
    collection_name: str = Field(default="nsrl_tech_docs", description="知识库集合名称")
    k: int = Field(default=15, ge=1, le=50, description="返回结果数量")
    title_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="标题权重")
    content_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="内容权重")


class SearchResponse(BaseModel):
    """搜索响应"""
    success: bool
    results: List[Dict[str, Any]]
    total: int
    message: Optional[str] = None


# ==================== 初始化函数 ====================

def get_vector_store(collection_name: str):
    """获取或创建向量存储"""
    if collection_name not in vector_stores:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        logger.info(f"初始化向量存储: {collection_name} (Qdrant: {qdrant_host}:{qdrant_port})")
        
        vector_store = embedding_init(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name
        )
        vector_stores[collection_name] = vector_store
    
    return vector_stores[collection_name]


def get_rag_tool(collection_name: str):
    """获取或创建 RAG 工具"""
    if collection_name not in rag_tools:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        
        logger.info(f"创建 RAG 工具: {collection_name}")
        
        rag_tool = create_rag_tool(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name
        )
        rag_tools[collection_name] = rag_tool
    
    return rag_tools[collection_name]


# ==================== MCP 协议端点 ====================

@app.get("/")
async def root():
    """根端点，返回服务信息"""
    return {
        "service": "MCP Knowledge Base Search Server",
        "version": "1.0.0",
        "protocol": "MCP",
        "endpoints": {
            "list_tools": "/mcp/tools",
            "call_tool": "/mcp/tools/call",
            "search": "/api/search"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 尝试连接默认知识库
        get_vector_store("nsrl_tech_docs")
        return {"status": "healthy", "qdrant": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/mcp/tools", response_model=MCPListToolsResponse)
async def list_tools():
    """
    MCP 协议：列出可用工具
    """
    tools = [
        MCPTool(
            name="knowledge_base_search",
            description="在知识库中搜索相关信息。支持语义搜索和混合搜索。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询语句"
                    },
                    "collection_name": {
                        "type": "string",
                        "description": "知识库集合名称，默认为 nsrl_tech_docs",
                        "default": "nsrl_tech_docs"
                    },
                    "k": {
                        "type": "integer",
                        "description": "返回结果数量，范围 1-50，默认 15",
                        "default": 15,
                        "minimum": 1,
                        "maximum": 50
                    },
                    "title_weight": {
                        "type": "number",
                        "description": "标题权重，范围 0.0-1.0，默认 0.6",
                        "default": 0.6,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "content_weight": {
                        "type": "number",
                        "description": "内容权重，范围 0.0-1.0，默认 0.4",
                        "default": 0.4,
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                },
                "required": ["query"]
            }
        )
    ]
    
    return MCPListToolsResponse(tools=tools)


@app.post("/mcp/tools/call", response_model=MCPCallToolResponse)
async def call_tool(request: MCPCallToolRequest):
    """
    MCP 协议：调用工具
    """
    if request.name != "knowledge_base_search":
        raise HTTPException(
            status_code=400,
            detail=f"未知工具: {request.name}"
        )
    
    # 提取参数
    query = request.arguments.get("query")
    if not query:
        raise HTTPException(
            status_code=400,
            detail="缺少必需参数: query"
        )
    
    collection_name = request.arguments.get("collection_name", "nsrl_tech_docs")
    k = request.arguments.get("k", 15)
    title_weight = request.arguments.get("title_weight", 0.6)
    content_weight = request.arguments.get("content_weight", 0.4)
    
    try:
        # 执行搜索
        vector_store = get_vector_store(collection_name)
        results = vector_store.weighted_hybrid_search(
            query=query,
            k=k,
            title_weight=title_weight,
            content_weight=content_weight
        )
        
        # 格式化结果
        formatted_results = []
        for i, (doc, score) in enumerate(results, 1):
            metadata = doc.metadata
            result_item = {
                "index": i,
                "score": float(score),
                "content": doc.page_content,
                "metadata": {
                    "title": metadata.get('title', metadata.get('title_text', '无标题')),
                    "source": metadata.get('source', '未知'),
                    "is_qa_pair": metadata.get('is_qa_pair', False) or metadata.get('type') == 'qa'
                }
            }
            formatted_results.append(result_item)
        
        # 构建响应
        response_text = format_search_results(formatted_results)
        
        return MCPCallToolResponse(
            content=[
                {
                    "type": "text",
                    "text": response_text
                }
            ],
            isError=False
        )
    
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}", exc_info=True)
        return MCPCallToolResponse(
            content=[
                {
                    "type": "text",
                    "text": f"搜索失败: {str(e)}"
                }
            ],
            isError=True
        )


# ==================== RESTful API 端点 ====================

@app.post("/api/search", response_model=SearchResponse)
async def search_knowledge_base(request: SearchRequest):
    """
    RESTful API：搜索知识库
    这是一个更简单的 API 接口，不遵循 MCP 协议
    """
    try:
        vector_store = get_vector_store(request.collection_name)
        
        results = vector_store.weighted_hybrid_search(
            query=request.query,
            k=request.k,
            title_weight=request.title_weight,
            content_weight=request.content_weight
        )
        
        # 格式化结果
        formatted_results = []
        for i, (doc, score) in enumerate(results, 1):
            metadata = doc.metadata
            result_item = {
                "index": i,
                "score": float(score),
                "content": doc.page_content,
                "metadata": {
                    "title": metadata.get('title', metadata.get('title_text', '无标题')),
                    "source": metadata.get('source', '未知'),
                    "is_qa_pair": metadata.get('is_qa_pair', False) or metadata.get('type') == 'qa'
                }
            }
            formatted_results.append(result_item)
        
        return SearchResponse(
            success=True,
            results=formatted_results,
            total=len(formatted_results)
        )
    
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"搜索失败: {str(e)}"
        )


# ==================== 辅助函数 ====================

def format_search_results(results: List[Dict[str, Any]]) -> str:
    """格式化搜索结果为文本"""
    if not results:
        return "未在知识库中找到相关信息。"
    
    formatted_lines = []
    qa_pairs_found = 0
    regular_docs_found = 0
    
    for result in results:
        metadata = result["metadata"]
        if metadata.get("is_qa_pair"):
            qa_pairs_found += 1
            formatted_lines.append(
                f"【QA对知识库 - 结果 #{result['index']} (相似度: {result['score']:.4f})】\n"
                f"来源: {metadata.get('source', '未知')}\n"
                f"内容: {result['content']}\n"
                "----------------------------------------"
            )
        else:
            regular_docs_found += 1
            formatted_lines.append(
                f"【文档片段 - 结果 #{result['index']} (相似度: {result['score']:.4f})】\n"
                f"标题: {metadata.get('title', '无标题')}\n"
                f"内容: {result['content']}\n"
                f"来源: {metadata.get('source', '未知')}\n"
                "----------------------------------------"
            )
    
    # 添加统计信息
    if qa_pairs_found > 0:
        formatted_lines.insert(0, f"📚 在QA对知识库中找到 {qa_pairs_found} 个相关问答对，{regular_docs_found} 个文档片段：\n")
    else:
        formatted_lines.insert(0, f"📚 在知识库中找到 {regular_docs_found} 个相关文档片段：\n")
    
    return "\n".join(formatted_lines)


# ==================== 启动服务器 ====================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_SERVER_PORT", "8001"))
    
    logger.info(f"启动 MCP 知识库搜索服务器...")
    logger.info(f"监听地址: {host}:{port}")
    logger.info(f"Qdrant 配置: {os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

