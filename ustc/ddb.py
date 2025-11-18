from enum import Enum
from fastapi import FastAPI, HTTPException, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, AsyncGenerator, TypedDict, Annotated
import uuid
import logging
from contextlib import asynccontextmanager, contextmanager
import os
import requests
import tempfile
import shutil
import operator
import re
import json
import asyncio
import gc
import torch
from email._header_value_parser import parse_message_id
from operator import add
from psycopg_pool import AsyncConnectionPool
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, FieldCondition, MatchValue
from oss2 import Auth, Bucket
from starlette.responses import StreamingResponse
from chunks2embedding import (
    embedding_init,
    upsert_md_file,
    upsert_qa_pair,
    delete_by_source,
    list_all_collections,
    get_collection_info
)
from pdf2md import pdf2md
from pdf import (
    register_user_document_tool,
    get_user_document_tool,
    get_user_document_tool_by_session,
    list_user_document_tools
)
from searxng_server import create_search_tool
import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# 创建专门的对话日志记录器
chat_logger = logging.getLogger("chat-flow")
chat_logger.setLevel(logging.INFO)
# 防止日志传播到父级logger，避免重复输出
chat_logger.propagate = False

# 创建对话日志文件处理器
import os
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
chat_file_handler = logging.FileHandler(os.path.join(log_dir, "chat_flow.log"), encoding='utf-8')
chat_file_handler.setFormatter(logging.Formatter("%(asctime)s [CHAT] %(message)s"))
chat_logger.addHandler(chat_file_handler)

# 可选：同时输出到控制台（如果你想要实时看到对话日志）
# chat_console_handler = logging.StreamHandler()
# chat_console_handler.setFormatter(logging.Formatter("%(asctime)s [CHAT] %(message)s"))
# chat_logger.addHandler(chat_console_handler)

# 创建统一的日志记录器
logger = logging.getLogger("unified-service")

# ======================
# 显存管理工具
# ======================
class GPUResourceManager:
    """管理GPU资源，确保同一时间只有一个模型在使用"""
    
    def __init__(self):
        self.lock = asyncio.Lock()
        self.current_model = None
        self.model_instances = {}
        
    async def acquire(self, model_type):
        """获取GPU资源并确保指定类型的模型已加载"""
        logger.info(f"尝试获取GPU资源以使用 {model_type} 模型...")
        await self.lock.acquire()
        logger.info(f"已获取GPU资源，准备使用 {model_type} 模型")
        
        try:
            # 如果已经有其他模型加载，先清理
            if self.current_model and self.current_model != model_type:
                await self.release()
            
            # 记录当前使用的模型类型
            self.current_model = model_type
            return self
        except Exception as e:
            logger.error(f"获取GPU资源失败: {str(e)}")
            self.lock.release()
            raise
            
    async def release(self):
        """释放GPU资源，清理当前模型"""
        if self.current_model:
            logger.info(f"正在清理 {self.current_model} 模型占用的资源...")
            # 从实例缓存中移除
            if self.current_model in self.model_instances:
                del self.model_instances[self.current_model]
            self.current_model = None
            
            # 显式清理GPU内存
            self.clear_gpu_memory()
            
        logger.info("GPU资源已释放")
        self.lock.release()
        
    def clear_gpu_memory(self):
        """清理GPU内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info(f"GPU内存已清理 - 已释放 {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    def get_ollama_model(self):
        """获取Ollama模型实例（按需加载）"""
        if "ollama" not in self.model_instances:
            inference_server_url = "http://localhost:11434/v1"
            self.model_instances["ollama"] = ChatOpenAI(
                model="qwen3:4b",
                openai_api_key="none",
                openai_api_base=inference_server_url,
                max_tokens=300,
                temperature=0.7,
                timeout=30.0,
                request_timeout=30.0,
                max_retries=2
            )
        return self.model_instances["ollama"]
    
    def get_embedding_model(self):
        """获取向量化模型（按需加载）"""
        if "embedding" not in self.model_instances:
            # 这里可以根据需要初始化向量化模型
            # 由于原始代码中没有显示具体实现，我们只返回初始化函数
            from chunks2embedding import embedding_init
            self.model_instances["embedding"] = embedding_init
        return self.model_instances["embedding"]
    
    def get_ocr_model(self):
        """获取OCR模型（按需加载）"""
        if "ocr" not in self.model_instances:
            # 由于原始代码中没有显示具体实现，我们只返回初始化函数
            from pdf2md import pdf2md
            self.model_instances["ocr"] = pdf2md
        return self.model_instances["ocr"]

# 创建全局GPU资源管理器
gpu_resource_manager = GPUResourceManager()

# ======================
# 全局共享资源
# ======================
qdrant_client = None
checkpointer = None
rag_tool_cache = {}
web_search_tool = None

# OSS 配置（从环境变量读取）
OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET", "")
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "https://oss-cn-hangzhou.aliyuncs.com")
OSS_BUCKET = os.getenv("OSS_BUCKET", "")

# 本地临时路径
LOCAL_DIR = "~/autodl-tmp/pdf"
os.makedirs(LOCAL_DIR, exist_ok=True)

# ======================
# 创建路由
# ======================
kb_router = APIRouter(prefix="/kb", tags=["知识库管理"])
agent_router = APIRouter(prefix="/agent", tags=["对话Agent"])

# ======================
# 共享工具函数
# ======================
def get_current_knowledge_base_info(kb_name: str):
    """只获取指定知识库的文档信息（不包含其他知识库）"""
    global qdrant_client
    try:
        # 使用全局Qdrant客户端
        if qdrant_client is None:
            qdrant_client = QdrantClient(host="localhost", port=6333)
        # 检查集合是否存在
        try:
            qdrant_client.get_collection(kb_name)
        except:
            return {
                "name": kb_name,
                "exists": False,
                "points_count": 0,
                "documents": [],
                "document_count": 0
            }
        # 获取知识库中的所有文档块（分页获取所有数据）
        all_points = []
        offset = None
        while True:
            # 使用scroll API获取数据
            points, next_offset = qdrant_client.scroll(
                collection_name=kb_name,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=offset
            )
            all_points.extend(points)
            if not next_offset:
                break
            offset = next_offset
        # 提取唯一文档名
        document_names = set()
        for point in all_points:
            try:
                # 尝试访问source字段
                if "metadata" in point.payload and "source" in point.payload["metadata"]:
                    document_names.add(point.payload["metadata"]["source"])
            except Exception as e:
                logger.warning(f"处理点时出错: {str(e)}")
        return {
            "name": kb_name,
            "exists": True,
            "points_count": len(all_points),
            "documents": list(document_names),
            "document_count": len(document_names)
        }
    except Exception as e:
        logger.error(f"获取知识库信息失败: {str(e)}", exc_info=True)
        return {
            "name": kb_name,
            "exists": False,
            "points_count": 0,
            "documents": [],
            "document_count": 0
        }

def download_pdf_from_oss(file_key, local_path):
    """从OSS下载文件"""
    try:
        # 初始化OSS客户端
        auth = Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
        bucket = Bucket(auth, OSS_ENDPOINT, OSS_BUCKET)
        bucket.get_object_to_file(file_key, local_path)
        logger.info(f"文件下载成功: {local_path}")
        return True
    except Exception as e:
        logger.error(f"文件下载失败: {str(e)}")
        return False

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

# ======================
# 知识库管理API路由
# ======================
# 1. 定义API请求模型
class KnowledgeBaseAction(str, Enum):
    CREATE = "create"
    UPLOAD = "upload"
    DELETE_DOCUMENT = "delete_document"
    DELETE = "delete"

class KnowledgeBaseRequest(BaseModel):
    action: KnowledgeBaseAction
    name: str
    document_name: Optional[str] = None

# 2. 修复KnowledgeBaseResponse定义
class KnowledgeBaseResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# 新增：问答对上传请求模型
class QAPairRequest(BaseModel):
    knowledge_base_name: str
    question: str
    answer: str
    document_name: Optional[str] = None  # 新增：文档名，如果不提供则自动生成
    metadata: Optional[Dict[str, Any]] = None  # 可选的元数据，如来源、标签等

# 6. 简化后的API端点
@kb_router.post("/api/knowledge-base", response_model=KnowledgeBaseResponse)
async def manage_knowledge_base(request: KnowledgeBaseRequest):
    """统一的知识库管理端点，只返回当前知识库信息"""
    global qdrant_client
    try:
        if request.action == KnowledgeBaseAction.CREATE:
            # 检查集合是否已存在
            if qdrant_client is None:
                qdrant_client = QdrantClient(host="localhost", port=6333)
            try:
                # 尝试获取集合，如果存在则返回exists
                qdrant_client.get_collection(request.name)
                status = "exists"
                message = f"知识库 '{request.name}' 已存在"
            except:
                # 集合不存在，创建它
                try:
                    qdrant_client.create_collection(
                        collection_name=request.name,
                        vectors_config={
                            "title": VectorParams(size=1024, distance=Distance.COSINE),
                            "content": VectorParams(size=1024, distance=Distance.COSINE)
                        }
                    )
                    status = "created"
                    message = f"知识库 '{request.name}' 创建成功"
                except Exception as e:
                    logger.error(f"创建知识库失败: {str(e)}")
                    return KnowledgeBaseResponse(
                        success=False,
                        message=f"创建知识库失败: {str(e)}",
                        data={"name": request.name}
                    )
            # 仅获取当前知识库的信息
            current_kb = get_current_knowledge_base_info(request.name)
            return KnowledgeBaseResponse(
                success=True,
                message=message,
                data={
                    "name": request.name,
                    "status": status,
                    "document_count": current_kb["document_count"],
                    "points_count": current_kb["points_count"],
                    "documents": current_kb["documents"]
                }
            )
        elif request.action == KnowledgeBaseAction.DELETE:
            try:
                if qdrant_client is None:
                    qdrant_client = QdrantClient(host="localhost", port=6333)
                qdrant_client.delete_collection(request.name)
                message = f"知识库 '{request.name}' 已删除"
                return KnowledgeBaseResponse(
                    success=True,
                    message=message,
                    data={
                        "name": request.name,
                        "document_count": 0,
                        "points_count": 0,
                        "documents": []
                    }
                )
            except Exception as e:
                logger.error(f"删除知识库失败: {str(e)}")
                return KnowledgeBaseResponse(
                    success=False,
                    message=f"删除知识库失败: {str(e)}",
                    data={"name": request.name}
                )
        elif request.action == KnowledgeBaseAction.UPLOAD:
            if not request.document_name:
                return KnowledgeBaseResponse(
                    success=False,
                    message="上传文档需要提供 document_name",
                    data={"name": request.name}
                )
            # 下载文件到临时目录
            oss_path = f"knowledge-documents/{request.name}/{request.document_name}.pdf"
            try:
                # 生成临时文件名
                orginal_file_name = f"{request.document_name}.pdf"
                local_input = os.path.join(LOCAL_DIR, orginal_file_name)
                if not download_pdf_from_oss(oss_path, local_input):
                    return KnowledgeBaseResponse(
                        success=False,
                        message="下载文件失败",
                        data={"name": request.name}
                    )
                
                # =============== 显存优化：使用OCR模型 ===============
                logger.info("开始使用OCR模型处理PDF文件...")
                await gpu_resource_manager.acquire("ocr")
                try:
                    # 下载文件
                    pdf2md_func = gpu_resource_manager.get_ocr_model()
                    pdf2md_func(local_input)
                finally:
                    await gpu_resource_manager.release()
                logger.info("OCR模型处理完成，资源已释放")
                
                tempfile = f'/home/easyai/OCRFlux/localworkspace/markdowns/{request.document_name}/{request.document_name}.md'
                
                # =============== 显存优化：使用向量化模型 ===============
                logger.info("开始使用向量化模型处理文档...")
                await gpu_resource_manager.acquire("embedding")
                try:
                    vector_store = gpu_resource_manager.get_embedding_model()(collection_name=request.name)
                    operation_info = upsert_md_file(tempfile, vector_store)
                finally:
                    await gpu_resource_manager.release()
                logger.info("向量化模型处理完成，资源已释放")
                
                # 仅获取当前知识库的信息
                current_kb = get_current_knowledge_base_info(request.name)
                return KnowledgeBaseResponse(
                    success=True,
                    message=f"文档 '{request.document_name}' 已上传到知识库 '{request.name}'",
                    data={
                        "name": request.name,
                        "document": request.document_name,
                        "details": str(operation_info),
                        "document_count": current_kb["document_count"],
                        "points_count": current_kb["points_count"],
                        "documents": current_kb["documents"]
                    }
                )
            finally:
                if os.path.exists(local_input):
                    os.remove(local_input)
                # 额外清理确保释放所有资源
                gpu_resource_manager.clear_gpu_memory()
        elif request.action == KnowledgeBaseAction.DELETE_DOCUMENT:
            if not request.document_name:
                return KnowledgeBaseResponse(
                    success=False,
                    message="删除文档需要提供 document_name",
                    data={"name": request.name}
                )
            
            # 直接调用您已有的函数
            # 不再自动添加.md后缀，因为QA对已经包含了.md后缀
            deletename = request.document_name
            
            # =============== 显存优化：使用向量化模型 ===============
            logger.info("开始使用向量化模型删除文档...")
            await gpu_resource_manager.acquire("embedding")
            try:
                vector_store = gpu_resource_manager.get_embedding_model()(collection_name=request.name)
                operation_info = delete_by_source(deletename, vector_store)
            finally:
                await gpu_resource_manager.release()
            logger.info("向量化模型操作完成，资源已释放")
            
            # 仅获取当前知识库的信息
            current_kb = get_current_knowledge_base_info(request.name)
            return KnowledgeBaseResponse(
                success=True,
                message=f"文档 '{request.document_name}' 已从知识库 '{request.name}' 中删除",
                data={
                    "name": request.name,
                    "document": request.document_name,
                    "details": str(operation_info),
                    "document_count": current_kb["document_count"],
                    "points_count": current_kb["points_count"],
                    "documents": current_kb["documents"]
                }
            )
    except Exception as e:
        logger.error(f"处理请求失败: {str(e)}", exc_info=True)
        return KnowledgeBaseResponse(
            success=False,
            message=f"服务器内部错误: {str(e)}",
            data={"action": request.action, "name": request.name}
        )

# 7. 简化其他端点
@kb_router.get("/api/knowledge-bases", response_model=KnowledgeBaseResponse)
async def list_knowledge_bases():
    """列出所有知识库 - 保持不变，因为这是另一个功能"""
    try:
        global qdrant_client
        if qdrant_client is None:
            qdrant_client = QdrantClient(host="localhost", port=6333)
        collections = qdrant_client.get_collections().collections
        kb_list = []
        for collection in collections:
            kb_info = get_current_knowledge_base_info(collection.name)
            kb_list.append({
                "name": collection.name,
                "points_count": kb_info["points_count"],
                "document_count": kb_info["document_count"]
            })
        return KnowledgeBaseResponse(
            success=True,
            message="知识库列表获取成功",
            data={
                "knowledge_bases": kb_list,
                "total": len(kb_list)
            }
        )
    except Exception as e:
        logger.error(f"获取知识库列表失败: {str(e)}", exc_info=True)
        return KnowledgeBaseResponse(
            success=False,
            message=f"获取知识库列表失败: {str(e)}",
            data={}
        )

@kb_router.get("/api/knowledge-base/{kb_name}/documents", response_model=KnowledgeBaseResponse)
async def list_documents(kb_name: str):
    """列出知识库中的文档 - 保持不变"""
    try:
        # 仅获取当前知识库的信息
        current_kb = get_current_knowledge_base_info(kb_name)
        if not current_kb["exists"]:
            return KnowledgeBaseResponse(
                success=False,
                message=f"知识库 '{kb_name}' 不存在",
                data={"name": kb_name}
            )
        return KnowledgeBaseResponse(
            success=True,
            message=f"知识库 '{kb_name}' 中的文档列表",
            data={
                "knowledge_base": kb_name,
                "documents": current_kb["documents"],
                "total": current_kb["document_count"],
                "points_count": current_kb["points_count"]
            }
        )
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}", exc_info=True)
        return KnowledgeBaseResponse(
            success=False,
            message=f"获取文档列表失败: {str(e)}",
            data={"name": kb_name}
        )

# 新增：问答对上传API端点
@kb_router.post("/api/qa-pair", response_model=KnowledgeBaseResponse)
async def upload_qa_pair(request: QAPairRequest):
    """上传问答对到指定知识库"""
    try:
        global qdrant_client
        # 检查知识库是否存在
        if qdrant_client is None:
            qdrant_client = QdrantClient(host="localhost", port=6333)
        
        try:
            qdrant_client.get_collection(request.knowledge_base_name)
        except:
            return KnowledgeBaseResponse(
                success=False,
                message=f"知识库 '{request.knowledge_base_name}' 不存在",
                data={"name": request.knowledge_base_name}
            )
        
        # =============== 显存优化：使用向量化模型 ===============
        logger.info("开始使用向量化模型处理问答对...")
        await gpu_resource_manager.acquire("embedding")
        try:
            vector_store = gpu_resource_manager.get_embedding_model()(collection_name=request.knowledge_base_name)
            
            # 构建问答对的文本内容
            qa_content = f"问题：{request.question}\n\n答案：{request.answer}"
            
            # 构建元数据
            metadata = {
                "source": "qa_pair",  # 保留原有标识，但会被upsert_qa_pair函数覆盖
                "type": "qa",
                "question": request.question,
                "answer": request.answer,
                "created_at": str(datetime.datetime.now())
            }
            # 如果提供了文档名，添加到metadata中
            if request.document_name:
                metadata["document_name"] = request.document_name
            if request.metadata:
                metadata.update(request.metadata)
            
            # 使用现有的upsert_md_file函数，但传入自定义内容
            # 注意：这里需要修改upsert_md_file函数以支持自定义内容，或者创建新的函数
            operation_info = upsert_qa_pair(qa_content, metadata, vector_store)
            
        finally:
            await gpu_resource_manager.release()
        logger.info("向量化模型处理完成，资源已释放")
        
        # 获取更新后的知识库信息
        current_kb = get_current_knowledge_base_info(request.knowledge_base_name)
        
        return KnowledgeBaseResponse(
            success=True,
            message=f"问答对已成功添加到知识库 '{request.knowledge_base_name}'",
            data={
                "name": request.knowledge_base_name,
                "qa_pair": {
                    "question": request.question,
                    "answer": request.answer
                },
                "details": str(operation_info),
                "document_count": current_kb["document_count"],
                "points_count": current_kb["points_count"],
                "documents": current_kb["documents"]
            }
        )
        
    except Exception as e:
        logger.error(f"上传问答对失败: {str(e)}", exc_info=True)
        return KnowledgeBaseResponse(
            success=False,
            message=f"上传问答对失败: {str(e)}",
            data={"name": request.knowledge_base_name}
        )

# 新增：批量问答对上传API端点
@kb_router.post("/api/qa-pairs/batch", response_model=KnowledgeBaseResponse)
async def upload_qa_pairs_batch(request: List[QAPairRequest]):
    """批量上传问答对到指定知识库"""
    try:
        global qdrant_client
        if not request:
            return KnowledgeBaseResponse(
                success=False,
                message="请求列表为空",
                data={}
            )
        
        # 检查知识库是否存在（使用第一个请求的知识库名称）
        knowledge_base_name = request[0].knowledge_base_name
        if qdrant_client is None:
            qdrant_client = QdrantClient(host="localhost", port=6333)
        
        try:
            qdrant_client.get_collection(knowledge_base_name)
        except:
            return KnowledgeBaseResponse(
                success=False,
                message=f"知识库 '{knowledge_base_name}' 不存在",
                data={"name": knowledge_base_name}
            )
        
        # =============== 显存优化：使用向量化模型 ===============
        logger.info(f"开始使用向量化模型批量处理 {len(request)} 个问答对...")
        await gpu_resource_manager.acquire("embedding")
        try:
            vector_store = gpu_resource_manager.get_embedding_model()(collection_name=knowledge_base_name)
            
            success_count = 0
            failed_count = 0
            failed_items = []
            
            for qa_request in request:
                try:
                    # 构建问答对的文本内容
                    qa_content = f"问题：{qa_request.question}\n\n答案：{qa_request.answer}"
                    
                    # 构建元数据
                    metadata = {
                        "source": "qa_pair",  # 保留原有标识，但会被upsert_qa_pair函数覆盖
                        "type": "qa",
                        "question": qa_request.question,
                        "answer": qa_request.answer,
                        "created_at": str(datetime.datetime.now())
                    }
                    # 如果提供了文档名，添加到metadata中
                    if qa_request.document_name:
                        metadata["document_name"] = qa_request.document_name
                    if qa_request.metadata:
                        metadata.update(qa_request.metadata)
                    
                    # 上传问答对
                    operation_info = upsert_qa_pair(qa_content, metadata, vector_store)
                    success_count += 1
                    
                except Exception as e:
                    failed_count += 1
                    failed_items.append({
                        "question": qa_request.question[:50] + "..." if len(qa_request.question) > 50 else qa_request.question,
                        "error": str(e)
                    })
                    logger.error(f"处理问答对失败: {str(e)}")
            
        finally:
            await gpu_resource_manager.release()
        logger.info("向量化模型批量处理完成，资源已释放")
        
        # 获取更新后的知识库信息
        current_kb = get_current_knowledge_base_info(knowledge_base_name)
        
        return KnowledgeBaseResponse(
            success=True,
            message=f"批量上传完成：成功 {success_count} 个，失败 {failed_count} 个",
            data={
                "name": knowledge_base_name,
                "total_requested": len(request),
                "success_count": success_count,
                "failed_count": failed_count,
                "failed_items": failed_items if failed_items else None,
                "document_count": current_kb["document_count"],
                "points_count": current_kb["points_count"],
                "documents": current_kb["documents"]
            }
        )
        
    except Exception as e:
        logger.error(f"批量上传问答对失败: {str(e)}", exc_info=True)
        return KnowledgeBaseResponse(
            success=False,
            message=f"批量上传问答对失败: {str(e)}",
            data={}
        )

# ======================
# 对话Agent API路由
# ======================
# 重要修复说明：
# 修复了双重累积问题，现在完全依赖 LangGraph 的 Annotated[list, add] 自动累积消息
# 避免了手动状态构建时重复添加消息导致的内存浪费和性能下降
# 修复前：messages = state.values["messages"] + [新消息] (手动累积)
# 修复后：messages = [新消息] (让 LangGraph 自动累积)
# ======================
# 3. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add]  # 消息历史，使用 LangGraph 的 add 操作符自动累积
    tool_call_count: Annotated[int, operator.add]  # 添加工具调用计数器
    knowledge_base_name: str  # 新增字段，用于存储当前会话使用的知识库名称
    user_document_tools: List[str]  # 新增字段，用于存储当前会话可用的用户文档工具名称
    web_search_enabled: bool  # 新增：记录web搜索是否启用

# 4. 修改模型调用节点
async def call_model(state: AgentState):
    """模型自主决策是否需要调用工具，包含参数验证和状态更新"""
    messages = state["messages"]
    knowledge_base_name = state.get("knowledge_base_name", "test")
    
    # =============== 新增：详细日志记录 ===============
    # 从消息中提取session信息（如果可用）
    session_info = ""
    if messages and isinstance(messages[-1], HumanMessage):
        # 尝试从消息内容中提取session信息
        last_message = messages[-1].content
        if "session_id:" in str(last_message):
            session_info = f" - 基于用户消息"
        chat_logger.info(f"🧠 模型开始思考")
        chat_logger.info(f"  📊 消息数量: {len(messages)}")
        chat_logger.info(f"  🔧 知识库: {knowledge_base_name}")
        chat_logger.info(f"  💭 用户问题: {last_message[:100]}{'...' if len(last_message) > 100 else ''}")
        if session_info:
            chat_logger.info(f"  ℹ️ {session_info}")
    # =============== 新增结束 ===============
    
    # 显存优化：限制会话长度，防止显存累积
    if len(messages) > 10:  # 限制会话历史长度
        # 保留最新的5条消息和系统提示
        messages = messages[-5:]
        logger.info("⚠️ 会话历史过长，已截断以节省显存")
        chat_logger.info(f"⚠️ 会话历史过长，已截断至 {len(messages)} 条消息")
    
    # 动态获取当前知识库的RAG工具
    rag_tool = get_rag_tool(knowledge_base_name)
    available_tools = [rag_tool]
    
    if state.get("web_search_enabled", True):
        global web_search_tool
        if web_search_tool is None:
            web_search_tool = create_search_tool()
        available_tools.append(web_search_tool)
    
    # 添加用户文档工具（如果有）
    user_document_tools_list = state.get("user_document_tools", [])
    for tool_name in user_document_tools_list:
        tool_info = get_user_document_tool(tool_name)
        if tool_info and "tool" in tool_info:
            available_tools.append(tool_info["tool"])
    
    chat_logger.info(f"🔧 可用工具: {[tool.name if hasattr(tool, 'name') else str(tool) for tool in available_tools]}")
    
    # =============== 显存优化：获取Ollama模型 ===============
    logger.info("准备使用Ollama模型处理对话...")
    chat_logger.info(f"🤖 获取Ollama模型...")
    await gpu_resource_manager.acquire("ollama")
    try:
        # 如果是初始查询，添加系统提示
        if len(messages) == 1 and isinstance(messages[0], HumanMessage):
            chat_logger.info(f"🆕 首次查询，添加系统提示")
            chat_logger.info(f"📚 当前知识库: {knowledge_base_name}")
            chat_logger.info(f"🌐 Web搜索状态: {'启用' if state.get('web_search_enabled', True) else '禁用'}")
            chat_logger.info(f"📄 用户文档工具数量: {len(user_document_tools_list)}")
            
            # 构建工具列表描述
            tools_description = f"""1. rag_knowledge_search: 查询内部知识库（主要包含医学相关内容）
            - 必须参数: query (string)
            - 当前知识库: {knowledge_base_name}
            - 调用示例: {{"name": "rag_knowledge_search", "arguments": {{"query": "NGS适用人群"}}}}
            - 特别说明: 知识库包含两种类型的内容：
              * QA对知识库: 以"问题：...答案：..."格式返回完整问答对，问题权重更高
              * 文档片段: 返回相关文档内容片段
            - 当检索到QA对时，系统会优先返回问题匹配度高的结果，并标记为"QA对知识库"
            2. web_search_tool: 查询最新互联网信息
            - 必须参数: query (string)
            - 调用示例: {{"name": "web_search_tool", "arguments": {{"query": "NGS最新研究"}}}}"""
            # 添加用户文档工具描述
            if user_document_tools_list:
                tools_description += "\n用户上传的文档搜索工具:"
                for tool_name in user_document_tools_list:
                    tool_info = get_user_document_tool(tool_name)
                    if tool_info:
                        tools_description += f"\n{tool_info['tool'].name}: {tool_info['tool'].description}"
                        tools_description += "\n   - 必须参数: query (string)"
            
            # 添加QA对说明
            tools_description += f"""
            3. 知识库内容说明:
            - 知识库 '{knowledge_base_name}' 包含PDF文档和问答对
            - PDF文档: 按原始文件名.md存储
            - 问答对: 按用户指定的文档名.md存储（如test.md）
            - 所有内容都可通过rag_knowledge_search统一搜索"""
            
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
                
                chat_logger.info(f"🔧 生成系统提示 (Web搜索禁用):")
                chat_logger.info(f"   📝 提示长度: {len(system_prompt)} 字符")
                chat_logger.info(f"   📋 提示内容: {system_prompt}")
            else:
                system_prompt = f"""你是一个医疗智能助手，可以使用以下工具：
{tools_description}
工作流程：
- 如果用户询问与其上传文档相关的内容，优先使用对应的文档搜索工具
- 对于一般医学问题，首先尝试使用 rag_knowledge_search
- 检查返回结果的最高相似度：
  * 如果最高相似度 ≥ 0.5，结果可能相关，可基于此生成回答
  * 如果最高相似度 < 0.5，结果不相关，应使用 web_search_tool
- 特别说明：当检索到QA对知识库内容时：
  * QA对的问题部分权重更高，问题匹配度高的结果会优先返回
  * 系统会返回完整的问答对内容，格式为"问题：...答案：..."
  * 这些QA对通常更准确、更直接地回答用户问题
  * 如果QA对知识库有相关内容，应优先基于QA对生成回答
确保最终回答整合所有可用信息
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
                
                chat_logger.info(f"🔧 生成系统提示 (Web搜索启用):")
                chat_logger.info(f"   📝 提示长度: {len(system_prompt)} 字符")
                chat_logger.info(f"   📋 提示内容: {system_prompt}")
            
            messages = [SystemMessage(content=system_prompt)] + messages
            chat_logger.info(f"✅ 系统提示已添加到消息列表")
            chat_logger.info(f"📊 添加系统提示后消息总数: {len(messages)}")
        
        # 检查工具调用次数 - 如果超过限制，强制模型提供答案
        current_tool_count = state.get("tool_call_count", 0)
        chat_logger.info(f"🔍 检查工具调用次数: 当前 {current_tool_count}/5")
        
        if current_tool_count >= 5:
            warning_msg = "⚠️ 重要提示：您已经调用了多次工具但仍未能提供最终答案。请基于已有信息立即提供完整回答，不要再调用工具。"
            messages.append(SystemMessage(content=warning_msg))
            chat_logger.warning(f"⚠️ 工具调用次数已达上限，添加强制回答提示")
            chat_logger.info(f"📝 添加的提示内容: {warning_msg}")
        
        # 计数web_search调用次数
        web_search_count = sum(1 for m in messages
                            if isinstance(m, AIMessage) and
                            m.tool_calls and
                            any(tc["name"] == "web_search_tool" for tc in m.tool_calls))
        
        chat_logger.info(f"🔍 检查Web搜索调用次数: 当前 {web_search_count}/5")
        
        # 检查是否已经调用过web_search但结果不理想
        if web_search_count >= 5:
            warning_msg = "⚠️ 重要提示：您已经多次使用网络搜索但仍未提供最终答案。请基于已有信息立即提供完整回答，不要再调用工具。"
            messages.append(SystemMessage(content=warning_msg))
            chat_logger.warning(f"⚠️ Web搜索调用次数已达上限，添加强制回答提示")
            chat_logger.info(f"📝 添加的提示内容: {warning_msg}")
        
        # === 关键新增：检查上一次工具调用结果的相似度 ===
        if len(messages) >= 3:
            last_tool_response = messages[-2]  # 上一条是工具响应
            chat_logger.info(f"🔍 检查上一次工具调用结果相似度...")
            
            if isinstance(last_tool_response, ToolMessage) and last_tool_response.content:
                chat_logger.info(f"📤 上一次工具响应: {last_tool_response.name}")
                chat_logger.info(f"📄 工具响应内容长度: {len(last_tool_response.content)} 字符")
                
                # 从工具响应中提取最高相似度
                highest_similarity = extract_highest_similarity(last_tool_response.content)
                chat_logger.info(f"🎯 提取的最高相似度: {highest_similarity:.4f}")
                
                # 如果最高相似度低于阈值，添加系统提示强制使用web_search_tool
                if highest_similarity < 0.5:
                    warning_msg = f"⚠️ 重要提示：本地知识库搜索结果相关性较低（最高相似度: {highest_similarity:.4f}）。请使用web_search_tool获取最新互联网信息。"
                    messages.append(SystemMessage(content=warning_msg))
                    logger.warning(f"检测到低相关性结果 (相似度: {highest_similarity:.4f})，强制使用网络搜索")
                    chat_logger.warning(f"⚠️ 检测到低相关性结果，添加强制网络搜索提示")
                    chat_logger.info(f"📝 添加的提示内容: {warning_msg}")
                else:
                    chat_logger.info(f"✅ 相似度 {highest_similarity:.4f} 达到阈值，无需强制网络搜索")
            else:
                chat_logger.info(f"ℹ️ 上一次消息不是工具响应或内容为空")
        else:
            chat_logger.info(f"ℹ️ 消息数量不足，跳过相似度检查")
        
        # 始终绑定所有可用工具
        chat_logger.info(f"🔧 开始绑定工具到模型...")
        chat_logger.info(f"📋 可用工具列表: {[tool.name if hasattr(tool, 'name') else str(tool) for tool in available_tools]}")
        
        model = gpu_resource_manager.get_ollama_model()
        chat_logger.info(f"🤖 获取到Ollama模型: {type(model).__name__}")
        
        model_with_tools = model.bind_tools(available_tools)
        chat_logger.info(f"✅ 工具绑定完成")
        
        # 调用模型
        chat_logger.info(f"🤖 开始调用Ollama模型...")
        chat_logger.info(f"📝 发送给模型的消息数量: {len(messages)}")
        
        # 记录发送给模型的详细消息内容
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                chat_logger.info(f"  📤 消息 {i+1} [用户]: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
            elif isinstance(msg, SystemMessage):
                chat_logger.info(f"  📤 消息 {i+1} [系统]: {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
            elif isinstance(msg, AIMessage):
                chat_logger.info(f"  📤 消息 {i+1} [AI]: {msg.content[:100] if msg.content else '无内容'}{'...' if msg.content and len(msg.content) > 100 else ''}")
            elif isinstance(msg, ToolMessage):
                chat_logger.info(f"  📤 消息 {i+1} [工具]: {msg.name} - {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
        
        response = await model_with_tools.ainvoke(messages)
        chat_logger.info(f"✅ 模型响应完成")
        
        # 记录模型响应的完整内容
        if hasattr(response, "content") and response.content:
            chat_logger.info(f"💬 模型回答完整内容:")
            chat_logger.info(f"   📄 回答长度: {len(response.content)} 字符")
            chat_logger.info(f"   📝 回答内容: {response.content}")
        else:
            chat_logger.info(f"⚠️ 模型响应无内容")
        
        # 记录工具调用的详细信息
        if hasattr(response, "tool_calls") and response.tool_calls:
            chat_logger.info(f"🔧 模型决定调用工具: {len(response.tool_calls)} 个")
            for i, tool_call in enumerate(response.tool_calls):
                chat_logger.info(f"  🔧 工具 {i+1} 详细信息:")
                chat_logger.info(f"    📛 工具名称: {tool_call['name']}")
                chat_logger.info(f"    🔑 工具ID: {tool_call['id']}")
                chat_logger.info(f"    📋 工具参数: {json.dumps(tool_call['args'], ensure_ascii=False, indent=2)}")
                chat_logger.info(f"    📊 参数数量: {len(tool_call['args'])}")
        else:
            chat_logger.info(f"💬 模型直接回答，无工具调用")
        
        # 记录模型响应的其他属性（避免访问Pydantic已弃用的属性）
        chat_logger.info(f"🔍 模型响应对象属性:")
        # 定义安全的属性列表，避免访问已弃用的Pydantic属性
        safe_attrs = ['content', 'tool_calls', 'response_metadata', 'usage_metadata', 'id', 'type']
        for attr in safe_attrs:
            if hasattr(response, attr):
                try:
                    value = getattr(response, attr)
                    if value is not None:
                        chat_logger.info(f"    📌 {attr}: {value}")
                except Exception as e:
                    chat_logger.info(f"    📌 {attr}: 无法获取值 ({str(e)})")
        
        # 关键修复：验证并清理工具调用参数
        if hasattr(response, "tool_calls") and response.tool_calls:
            chat_logger.info(f"🧹 开始清理工具调用参数...")
            chat_logger.info(f"📊 原始工具调用数量: {len(response.tool_calls)}")
            
            cleaned_tool_calls = []
            for i, tool_call in enumerate(response.tool_calls):
                chat_logger.info(f"  🔍 处理工具调用 {i+1}:")
                chat_logger.info(f"    📛 原始名称: {tool_call['name']}")
                chat_logger.info(f"    🔑 原始ID: {tool_call['id']}")
                chat_logger.info(f"    📋 原始参数: {json.dumps(tool_call['args'], ensure_ascii=False, indent=4)}")
                
                # 只保留有效的参数
                valid_args = {}
                # 根据工具名称处理参数
                if tool_call["name"] == "rag_knowledge_search":
                    # 仅保留query参数
                    if "query" in tool_call["args"]:
                        valid_args["query"] = tool_call["args"]["query"]
                        chat_logger.info(f"    ✅ 保留有效参数: query = {tool_call['args']['query']}")
                    else:
                        # 如果没有query参数，使用第一个参数或整个内容作为查询
                        first_arg = next(iter(tool_call["args"].values()), "未知查询")
                        valid_args["query"] = str(first_arg)
                        logger.warning(f"rag_knowledge_search缺少query参数，使用备用参数: {first_arg}")
                        chat_logger.warning(f"⚠️ rag_knowledge_search缺少query参数，使用备用参数: {first_arg}")
                elif tool_call["name"] == "web_search_tool":
                    # 仅保留query参数
                    if "query" in tool_call["args"]:
                        valid_args["query"] = tool_call["args"]["query"]
                        chat_logger.info(f"    ✅ 保留有效参数: query = {tool_call['args']['query']}")
                    else:
                        first_arg = next(iter(tool_call["args"].values()), "未知查询")
                        valid_args["query"] = str(first_arg)
                        logger.warning(f"web_search_tool缺少query参数，使用备用参数: {first_arg}")
                        chat_logger.warning(f"⚠️ web_search_tool缺少query参数，使用备用参数: {first_arg}")
                # 添加用户文档工具参数处理
                elif tool_call["name"].startswith("search_"):
                    # 仅保留query参数
                    if "query" in tool_call["args"]:
                        valid_args["query"] = tool_call["args"]["query"]
                        chat_logger.info(f"    ✅ 保留有效参数: query = {tool_call['args']['query']}")
                    else:
                        first_arg = next(iter(tool_call["args"].values()), "未知查询")
                        valid_args["query"] = str(first_arg)
                        logger.warning(f"{tool_call['name']}缺少query参数，使用备用参数: {first_arg}")
                        chat_logger.warning(f"⚠️ {tool_call['name']}缺少query参数，使用备用参数: {first_arg}")
                
                # 创建清理后的工具调用
                cleaned_tool_call = {
                    "name": tool_call["name"],
                    "args": valid_args,
                    "id": tool_call["id"]
                }
                cleaned_tool_calls.append(cleaned_tool_call)
                
                chat_logger.info(f"    🧹 清理后参数: {json.dumps(valid_args, ensure_ascii=False, indent=4)}")
                chat_logger.info(f"    📊 参数清理: {len(tool_call['args'])} -> {len(valid_args)}")
            
            # 替换原始的tool_calls
            response.tool_calls = cleaned_tool_calls
            logger.info(f"已清理工具调用参数，移除无效参数")
            chat_logger.info(f"✅ 工具调用参数清理完成")
            chat_logger.info(f"📊 清理后工具调用数量: {len(cleaned_tool_calls)}")
        else:
            chat_logger.info(f"ℹ️ 无需清理工具调用参数（无工具调用）")
        
        # 计算工具调用增量
        tool_call_increment = 1 if (hasattr(response, "tool_calls") and response.tool_calls) else 0
        
        # =============== 新增：详细日志记录 ===============
        chat_logger.info(f"📊 工具调用统计 - 本次增量: {tool_call_increment}")
        chat_logger.info(f"🎯 模型思考完成，准备返回结果")
        # =============== 新增结束 ===============
        return {
            "messages": [response],
            "tool_call_count": tool_call_increment,
            "knowledge_base_name": knowledge_base_name,  # 确保传递知识库名称
            "user_document_tools": user_document_tools_list,  # 确保传递用户文档工具列表
            "web_search_enabled": state.get("web_search_enabled", True)
        }
    finally:
        await gpu_resource_manager.release()
        logger.info("Ollama模型处理完成，资源已释放")
        chat_logger.info(f"🧹 GPU资源已释放")

# 5. 修改条件函数
def should_continue(state: AgentState):
    """决定是否需要调用工具或结束"""
    messages = state["messages"]
    last_message = messages[-1]
    tool_call_count = state.get("tool_call_count", 0)
    
    # =============== 新增：详细日志记录 ===============
    chat_logger.info(f"🤔 决策是否继续 - 工具调用次数: {tool_call_count}")
    # =============== 新增结束 ===============
    
    # 检查工具调用次数 - 超过5次强制结束
    if tool_call_count >= 5:
        chat_logger.info(f"🛑 工具调用次数已达上限({tool_call_count})，结束对话")
        return END
    # 如果有工具调用，则继续
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        chat_logger.info(f"🔄 检测到工具调用，继续执行工具节点")
        return "tools"
    # 否则结束
    chat_logger.info(f"✅ 无工具调用，对话结束")
    return END

# 6. 全局变量存储编译后的图
graph = None

def tool_node(state: AgentState):
    """自定义工具节点，能根据知识库名称动态获取工具"""
    messages = state["messages"]
    last_message = messages[-1]
    knowledge_base_name = state.get("knowledge_base_name", "test")
    
    # =============== 新增：详细日志记录 ===============
    chat_logger.info(f"🔧 开始执行工具节点")
    chat_logger.info(f"  🔧 知识库: {knowledge_base_name}")
    chat_logger.info(f"  📝 需要执行的工具调用: {len(last_message.tool_calls)} 个")
    # 尝试从状态中提取更多会话信息
    if hasattr(state, 'get') and state.get('session_id'):
        chat_logger.info(f"  🆔 Session ID: {state.get('session_id')}")
    if hasattr(state, 'get') and state.get('message_id'):
        chat_logger.info(f"  📨 Message ID: {state.get('message_id')}")
    
    for i, tool_call in enumerate(last_message.tool_calls):
        chat_logger.info(f"  🔧 工具 {i+1}: {tool_call['name']} - 参数: {tool_call['args']}")
    # =============== 新增结束 ===============
    
    # 动态获取当前知识库的RAG工具
    rag_tool = get_rag_tool(knowledge_base_name)
    # 创建工具映射
    tools = {
        "rag_knowledge_search": rag_tool
    }
    # 仅当启用时才添加web搜索工具
    if state.get("web_search_enabled", True):
        global web_search_tool
        if web_search_tool is None:
            web_search_tool = create_search_tool()
        tools["web_search_tool"] = web_search_tool
    # 添加用户文档工具
    user_document_tools_list = state.get("user_document_tools", [])
    for tool_name in user_document_tools_list:
        tool_info = get_user_document_tool(tool_name)
        if tool_info and "tool" in tool_info:
            tools[tool_info["tool"].name] = tool_info["tool"]
    chat_logger.info(f"🔧 可用工具: {list(tools.keys())}")
    
        # 执行所有工具调用
    outputs = []
    for i, tool_call in enumerate(last_message.tool_calls):
        tool_name = tool_call["name"]
        chat_logger.info(f"🚀 开始执行工具 {i+1}: {tool_name}")
        
        if tool_name in tools:
            tool = tools[tool_name]
            try:
                # 调用工具
                chat_logger.info(f"🔧 调用工具 {tool_name}")
                chat_logger.info(f"  📋 工具参数: {tool_call['args']}")
                chat_logger.info(f"  📋 工具参数详情: {json.dumps(tool_call['args'], ensure_ascii=False, indent=4)}")
                
                # 记录工具调用开始时间
                import time
                start_time = time.time()
                chat_logger.info(f"⏱️ 工具 {tool_name} 开始执行...")
                
                response = tool.invoke(tool_call["args"])
                
                # 记录工具执行时间
                end_time = time.time()
                execution_time = end_time - start_time
                chat_logger.info(f"⏱️ 工具 {tool_name} 执行完成，耗时: {execution_time:.3f} 秒")
                
                # 记录工具返回结果
                response_str = str(response)
                chat_logger.info(f"✅ 工具 {tool_name} 执行成功")
                chat_logger.info(f"📤 工具返回结果长度: {len(response_str)} 字符")
                chat_logger.info(f"📄 工具返回完整内容:")
                chat_logger.info(f"   {response_str}")
                
                # 添加会话上下文信息
                chat_logger.info(f"  📊 工具执行统计:")
                chat_logger.info(f"    ⏱️ 执行时间: {execution_time:.3f} 秒")
                chat_logger.info(f"    📏 返回内容长度: {len(response_str)} 字符")
                chat_logger.info(f"    🎯 工具名称: {tool_name}")
                
                # 如果是RAG工具，记录相似度信息
                if tool_name == "rag_knowledge_search" and "相似度:" in response_str:
                    similarities = re.findall(r"相似度: ([\d.]+)", response_str)
                    if similarities:
                        similarities_float = [float(s) for s in similarities]
                        max_sim = max(similarities_float)
                        min_sim = min(similarities_float)
                        avg_sim = sum(similarities_float) / len(similarities_float)
                        chat_logger.info(f"🎯 RAG工具相似度统计:")
                        chat_logger.info(f"   📊 最高相似度: {max_sim:.4f}")
                        chat_logger.info(f"   📊 最低相似度: {min_sim:.4f}")
                        chat_logger.info(f"   📊 平均相似度: {avg_sim:.4f}")
                        chat_logger.info(f"   📊 相似度数量: {len(similarities)}")
                        chat_logger.info(f"   📋 所有相似度: {similarities}")
                
                # 记录工具响应的其他属性（避免访问Pydantic已弃用的属性）
                chat_logger.info(f"🔍 工具响应对象属性:")
                # 定义安全的属性列表，避免访问已弃用的Pydantic属性
                safe_attrs = ['content', 'name', 'status', 'tool_call_id']
                for attr in safe_attrs:
                    if hasattr(response, attr):
                        try:
                            value = getattr(response, attr)
                            if value is not None:
                                chat_logger.info(f"    📌 {attr}: {value}")
                        except Exception as e:
                            chat_logger.info(f"    📌 {attr}: 无法获取值 ({str(e)})")
                
                outputs.append(
                    ToolMessage(
                        content=str(response),
                        name=tool_name,
                        tool_call_id=tool_call["id"]
                    )
                )
            except Exception as e:
                chat_logger.error(f"❌ 工具 {tool_name} 执行失败: {str(e)}")
                chat_logger.error(f"🔍 错误详情: {type(e).__name__}: {str(e)}")
                
                # 记录详细的错误堆栈
                import traceback
                error_traceback = traceback.format_exc()
                chat_logger.error(f"📚 错误堆栈: {error_traceback}")
                
                outputs.append(
                    ToolMessage(
                        content=f"工具调用错误: {str(e)}",
                        name=tool_name,
                        status="error",
                        tool_call_id=tool_call["id"]
                    )
                )
        else:
            chat_logger.error(f"❌ 工具 {tool_name} 不存在，可用工具: {list(tools.keys())}")
            outputs.append(
                ToolMessage(
                    content=f"Error: {tool_name} is not a valid tool, try one of [{', '.join(tools.keys())}]",
                    name=tool_name,
                    status="error"
                )
            )
    
    chat_logger.info(f"✅ 工具节点执行完成，返回 {len(outputs)} 个结果")
    return {"messages": outputs}

# 8. 核心API端点
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

@agent_router.post(
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
    
    # =============== 新增：详细日志记录 ===============
    chat_logger.info(f"🚀 开始处理聊天请求")
    chat_logger.info(f"  🆔 Session ID: {session_id}")
    if request.message_id:
        chat_logger.info(f"  📨 Message ID: {request.message_id}")
    chat_logger.info(f"  📝 用户输入: {request.message}")
    chat_logger.info(f"  🔧 知识库: {request.knowledge_base_name}")
    chat_logger.info(f"  🌐 Web搜索: {'启用' if request.enable_web_search else '禁用'}")
    if request.url:
        chat_logger.info(f"  📄 文档URL: {request.url}")
    chat_logger.info(f"  ⏰ 请求时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # =============== 新增结束 ===============
    
    try:
        # 获取当前会话状态
        state = await graph.aget_state(config)
        # 构建新状态
        if state is None or not isinstance(state.values, dict) or "messages" not in state.values:
            # 新会话（包括状态不完整的情况）
            chat_logger.info(f"🆕 创建新会话")
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
                    chat_logger.info(f"✅ 注册文档工具: {tool_name}")
                except Exception as e:
                    logger.error(f"文档处理失败: {str(e)}")
                    chat_logger.error(f"❌ 文档处理失败: {str(e)}")
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
            # 续会话 - 只添加新消息，让 LangGraph 自动管理状态累积
            # 修复说明：避免双重累积问题，让 LangGraph 的 Annotated[list, add] 自动处理消息历史
            chat_logger.info(f"🔄 继续现有会话，历史消息数: {len(state.values.get('messages', []))}")
            chat_logger.info(f"🔧 修复：使用 LangGraph 自动累积，避免手动重复添加消息")
            # 保留之前的知识库名称，即使请求中提供了新值（避免中途切换知识库导致混淆）
            knowledge_base_name = state.values.get("knowledge_base_name", request.knowledge_base_name)
            # =============== 新增：处理文档URL（如果是新上传的文档） ===============
            user_document_tools_list = state.values.get("user_document_tools", [])
            chat_logger.info(f"📚 现有文档工具: {user_document_tools_list}")
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
                    chat_logger.info(f"✅ 新增文档工具: {tool_name}")
                except Exception as e:
                    logger.error(f"文档处理失败: {str(e)}")
                    chat_logger.error(f"❌ 新增文档失败: {str(e)}")
            # =============== 新增结束 ===============
            # 保留之前的web搜索设置，除非请求中提供了新值
            web_search_enabled = state.values.get("web_search_enabled", request.enable_web_search)
            initial_state = {
                "messages": [HumanMessage(content=request.message)],  # 修复：只添加新消息，让 LangGraph 自动累积
                # 重要说明：这里只添加新消息，LangGraph 的 Annotated[list, add] 会自动将新消息
                # 与之前保存的状态中的消息列表合并，避免手动重复累积导致的内存浪费
                "knowledge_base_name": knowledge_base_name,
                "tool_call_count": state.values.get("tool_call_count", 0),
                "user_document_tools": user_document_tools_list,
                "web_search_enabled": web_search_enabled
            }
        chat_logger.info(f"🎯 初始状态构建完成，工具数量: {len(initial_state.get('user_document_tools', []))}")
        
        # 执行对话流
        chat_logger.info(f"🔄 开始执行对话流程...")
        final_state = None
        async for step in graph.astream(initial_state, config=config, stream_mode="values"):
            final_state = step
        if not final_state:
            chat_logger.error(f"❌ 对话流程未产生有效响应")
            raise HTTPException(
                status_code=500,
                detail="对话流程未产生有效响应"
            )
        
        chat_logger.info(f"✅ 对话流程执行完成")
        # 提取最新回复
        last_msg = final_state["messages"][-1]
        if not isinstance(last_msg, AIMessage):
            chat_logger.error(f"❌ 无效的模型响应类型: {type(last_msg)}")
            raise HTTPException(
                status_code=500,
                detail="无效的模型响应类型"
            )
        
        # 记录最终状态的详细信息
        chat_logger.info(f"📊 最终状态详细信息:")
        chat_logger.info(f"  🆔 Session ID: {session_id}")
        if request.message_id:
            chat_logger.info(f"  📨 Message ID: {request.message_id}")
        chat_logger.info(f"  📝 总消息数量: {len(final_state['messages'])}")
        chat_logger.info(f"  🔧 工具调用计数: {final_state.get('tool_call_count', 0)}")
        chat_logger.info(f"  📚 知识库名称: {final_state.get('knowledge_base_name', 'unknown')}")
        chat_logger.info(f"  🌐 Web搜索启用: {final_state.get('web_search_enabled', False)}")
        chat_logger.info(f"  📄 用户文档工具: {final_state.get('user_document_tools', [])}")
        if request.url:
            chat_logger.info(f"  📄 文档URL: {request.url}")
        
        # 记录所有消息的详细信息
        chat_logger.info(f"📋 完整对话历史:")
        for i, msg in enumerate(final_state["messages"]):
            if isinstance(msg, HumanMessage):
                chat_logger.info(f"  📤 消息 {i+1} [用户]: {msg.content}")
            elif isinstance(msg, AIMessage):
                chat_logger.info(f"  📤 消息 {i+1} [AI]: {msg.content}")
            elif isinstance(msg, SystemMessage):
                chat_logger.info(f"  📤 消息 {i+1} [系统]: {msg.content}")
            elif isinstance(msg, ToolMessage):
                chat_logger.info(f"  📤 消息 {i+1} [工具 {msg.name}]: {msg.content}")
        
        # 收集工具调用信息（用于调试）
        tool_calls = []
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            chat_logger.info(f"🔧 检测到工具调用: {len(last_msg.tool_calls)} 个")
            for i, tool_call in enumerate(last_msg.tool_calls):
                tool_calls.append({
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                    "id": tool_call["id"]
                })
                chat_logger.info(f"  🔧 工具 {i+1}: {tool_call['name']} - 参数: {tool_call['args']}")
        else:
            chat_logger.info(f"💬 无工具调用，直接回答")
        
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
        
        # 记录最终返回的详细信息
        chat_logger.info(f"📤 返回最终回答:")
        chat_logger.info(f"  🆔 Session ID: {session_id}")
        if request.message_id:
            chat_logger.info(f"  📨 Message ID: {request.message_id}")
        chat_logger.info(f"  📄 回答长度: {len(last_msg.content)} 字符")
        chat_logger.info(f"  📝 回答内容: {last_msg.content}")
        chat_logger.info(f"  📊 历史记录数量: {len(history)}")
        chat_logger.info(f"  🔧 工具调用数量: {len(tool_calls) if tool_calls else 0}")
        if request.url:
            chat_logger.info(f"  📄 文档URL: {request.url}")
        
        chat_logger.info(f"🎯 对话完成")
        chat_logger.info(f"  🆔 Session ID: {session_id}")
        if request.message_id:
            chat_logger.info(f"  📨 Message ID: {request.message_id}")
        chat_logger.info(f"  ⏰ 完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
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
@agent_router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    流式响应聊天请求（固定分块大小）
    - 使用标准SSE格式
    - 修复流式响应问题
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
            # 获取当前会话状态和构建初始状态
            state = await graph.aget_state(config)
            if state is None or not isinstance(state.values, dict) or "messages" not in state.values:
                # 新会话处理...
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
                # 续会话处理...
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
                    "messages": [HumanMessage(content=request.message)],  # 修复：只添加新消息，让 LangGraph 自动累积
                    # 重要说明：这里只添加新消息，LangGraph 的 Annotated[list, add] 会自动将新消息
                    # 与之前保存的状态中的消息列表合并，避免手动重复累积导致的内存浪费
                    "knowledge_base_name": knowledge_base_name,
                    "tool_call_count": state.values.get("tool_call_count", 0),
                    "user_document_tools": user_document_tools_list,
                    "web_search_enabled": web_search_enabled
                }
            
            # 优化分块大小和流式处理逻辑
            CHUNK_SIZE = 50  # 增大分块大小，减少过度分割
            full_text = ""
            last_sent_length = 0  # 记录已发送的文本长度
            
            logger.info(f" 开始流式响应，分块大小: {CHUNK_SIZE}")
            
            # 发送开始标记
            start_data = {
                "text": "",
                "finish_reason": "start",
                "session_id": session_id,
                "message_id": message_id
            }
            yield f"data: {json.dumps(start_data)}\n"
            await asyncio.sleep(0.01)
            
            # 记录流式响应开始
            logger.info(f"🔄 流式响应开始")
            logger.info(f"  🆔 Session ID: {session_id}")
            if message_id:
                logger.info(f"  📨 Message ID: {message_id}")
            logger.info(f"  📝 用户消息: {request.message[:100]}{'...' if len(request.message) > 100 else ''}")
            if request.url:
                logger.info(f"  📄 文档URL: {request.url}")
            logger.info(f"  ⏰ 开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                async for step in graph.astream_log(initial_state, config=config):
                    if isinstance(step, dict) and "ops" in step:
                        ops = step["ops"]
                    elif hasattr(step, "ops"):
                        ops = step.ops
                    else:
                        continue
                    
                    for op in ops:
                        path = op.get("path", "") if isinstance(op, dict) else getattr(op, "path", "")
                        value = op.get("value") if isinstance(op, dict) else getattr(op, "value", None)
                        
                        # 检查是否是模型调用的日志
                        if path.startswith("/logs/call_model/") and value is not None:
                            if isinstance(value, dict) and "messages" in value:
                                for msg in value["messages"]:
                                    if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
                                        # 累积新内容
                                        new_content = msg.content
                                        if new_content:
                                            full_text += new_content
                                            logger.debug(f" 累积内容，当前长度: {len(full_text)}")
                                            
                                            # 检查是否有新内容需要发送
                                            if len(full_text) > last_sent_length:
                                                # 发送新增的内容
                                                new_chunk = full_text[last_sent_length:]
                                                logger.debug(f" 准备发送新内容，长度: {len(new_chunk)}")
                                                
                                                # 如果新内容超过分块大小，进行分块
                                                while len(new_chunk) > CHUNK_SIZE:
                                                    chunk = new_chunk[:CHUNK_SIZE]
                                                    new_chunk = new_chunk[CHUNK_SIZE:]
                                                    
                                                    # 发送分块内容
                                                    data = {
                                                        "text": chunk,
                                                        "finish_reason": None,
                                                        "session_id": session_id,
                                                        "message_id": message_id
                                                    }
                                                    yield f"data: {json.dumps(data)}\n"
                                                    await asyncio.sleep(0.01)
                                                    logger.debug(f" 发送分块内容，长度: {len(chunk)}")
                                                
                                                # 发送剩余的新内容
                                                if new_chunk:
                                                    data = {
                                                        "text": new_chunk,
                                                        "finish_reason": None,
                                                        "session_id": session_id,
                                                        "message_id": message_id
                                                    }
                                                    yield f"data: {json.dumps(data)}\n"
                                                    await asyncio.sleep(0.01)
                                                    logger.debug(f" 发送剩余内容，长度: {len(new_chunk)}")
                                                
                                                # 更新已发送长度
                                                last_sent_length = len(full_text)
                                                logger.debug(f"✅ 已发送长度更新为: {last_sent_length}")
                                                
                                                # 定期清理显存
                                                if len(full_text) % 200 == 0:
                                                    import gc
                                                    gc.collect()
                                                    logger.info("粒 执行显存清理")
                                                
                                                # 发送进度指示器
                                                if len(full_text) % 100 == 0:
                                                    progress_data = {
                                                        "text": "",
                                                        "finish_reason": "progress",
                                                        "session_id": session_id,
                                                        "message_id": message_id,
                                                        "progress": {
                                                            "total_length": len(full_text),
                                                            "sent_length": last_sent_length
                                                        }
                                                    }
                                                    yield f"data: {json.dumps(progress_data)}\n"
                                                    await asyncio.sleep(0.01)
                
                # 然后获取最终状态，确保对话流程完成
                logger.info(" 日志流完成，获取最终状态...")
                final_state = None
                async for step in graph.astream(initial_state, config=config, stream_mode="values"):
                    final_state = step
                
                # 如果有最终状态，发送最终回答
                if final_state and "messages" in final_state:
                    last_msg = final_state["messages"][-1]
                    if isinstance(last_msg, AIMessage) and hasattr(last_msg, "content") and last_msg.content:
                        final_content = last_msg.content
                        # 检查是否与已发送内容不同
                        if final_content != full_text:
                            logger.info(f" 发送最终回答，长度: {len(final_content)}")
                            # 发送最终回答
                            data = {
                                "text": final_content,
                                "finish_reason": "final_answer",
                                "session_id": session_id,
                                "message_id": message_id
                            }
                            yield f"data: {json.dumps(data)}\n"
                            await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"❌ 流式响应失败: {e}")
                # 显存清理
                import gc
                gc.collect()
                raise e
            
            # 发送剩余内容（如果有）
            if full_text and len(full_text) > last_sent_length:
                remaining_text = full_text[last_sent_length:]
                if remaining_text:
                    data = {
                        "text": remaining_text,
                        "finish_reason": None,
                        "session_id": session_id,
                        "message_id": message_id
                    }
                    yield f"data: {json.dumps(data)}\n"
                    await asyncio.sleep(0.01)
                    logger.debug(f" 发送最终剩余内容，长度: {len(remaining_text)}")
            
            # 发送结束标记
            end_data = {
                "text": "",
                "finish_reason": "stop",
                "session_id": session_id,
                "message_id": message_id
            }
            yield f"data: {json.dumps(end_data)}\n"
            
            # 记录流式响应完成
            logger.info(f"✅ 流式响应完成")
            logger.info(f"  🆔 Session ID: {session_id}")
            if message_id:
                logger.info(f"  📨 Message ID: {message_id}")
            logger.info(f"  📊 总内容长度: {len(full_text)}")
            logger.info(f"  📤 已发送长度: {last_sent_length}")
            logger.info(f"  ⏰ 完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 最终显存清理
            import gc
            gc.collect()
            logger.info(f"粒 流式响应完成，执行最终显存清理。总内容长度: {len(full_text)}, 已发送长度: {last_sent_length}")
        except Exception as e:
            logger.error(f"❌ event_generator 失败: {e}")
            # 发送错误信息
            error_data = {
                "text": f"错误: {str(e)}",
                "finish_reason": "error",
                "session_id": session_id,
                "message_id": message_id
            }
            yield f"data: {json.dumps(error_data)}\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"  # 确保这是正确的媒体类型
    )


# 10. 会话管理API
@agent_router.get("/sessions/{session_id}")
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

@agent_router.delete("/sessions/{session_id}")
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

# ======================
# 统一健康检查和主应用
# ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant_client, checkpointer, web_search_tool, graph
    try:
        # 1. 初始化Qdrant客户端（知识库管理需要）
        logger.info("初始化Qdrant客户端...")
        qdrant_client = QdrantClient(host="localhost", port=6333)
        logger.info("✅ Qdrant客户端初始化成功")
        
        # 2. 初始化数据库连接池（Agent需要）
        logger.info("初始化数据库连接池...")
        DB_URI = os.getenv(
            "DB_URI",
            "postgresql://postgres:postgres@localhost:5432/langgraph_db?sslmode=disable"
        )
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        pool = AsyncConnectionPool(
            conninfo=DB_URI,
            max_size=20,
            kwargs=connection_kwargs
        )
        await pool.open()
        logger.info("初始化数据库检查点存储...")
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        logger.info("✅ 数据库检查点存储初始化成功")
        
        # 3. 初始化web搜索工具
        try:
            logger.info("初始化web搜索工具...")
            web_search_tool = create_search_tool()
            logger.info("✅ 工具加载成功: web_search")
        except Exception as e:
            logger.error(f"❌ 工具加载失败: {str(e)}")
            # 不抛出异常，继续初始化其他组件
        
        # 4. 编译Agent图
        logger.info("编译Agent图...")
        builder = StateGraph(AgentState)
        builder.add_node("call_model", call_model)
        builder.add_node("tools", tool_node)
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
        # 清理资源
        if 'pool' in locals():
            await pool.close()
            logger.info("✅ 数据库连接池已关闭")

# 创建统一的FastAPI应用
app = FastAPI(
    title="统一知识库与对话Agent服务",
    description="整合知识库管理和对话Agent功能",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册子应用
app.include_router(kb_router)
app.include_router(agent_router)

# 健康检查端点（统一）
@app.get("/health")
async def health_check():
    """统一的健康检查"""
    try:
        # 检查Qdrant
        global qdrant_client
        kb_status = "disconnected"
        total_kb = 0
        if qdrant_client:
            try:
                collections = qdrant_client.get_collections().collections
                kb_status = "connected"
                total_kb = len(collections)
            except Exception as e:
                kb_status = f"error: {str(e)}"
        
        # 检查数据库
        db_status = "connected" if checkpointer else "disconnected"
        
        # 检查Agent图
        agent_status = "initialized" if graph else "not_initialized"
        
        return {
            "status": "healthy",
            "knowledge_base": kb_status,
            "agent_database": db_status,
            "agent_status": agent_status,
            "total_knowledge_bases": total_kb,
            "service_info": {
                "port": 8000,
                "kb_api_prefix": "/kb",
                "agent_api_prefix": "/agent"
            }
        }
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "service_info": {
                "port": 8000,
                "kb_api_prefix": "/kb",
                "agent_api_prefix": "/agent"
            }
        }

if __name__ == "__main__":
    import uvicorn
    logger.info("启动统一服务...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")