from enum import Enum
from fastapi import FastAPI, HTTPException, status, APIRouter, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
import time
from email._header_value_parser import parse_message_id
from operator import add
from psycopg_pool import AsyncConnectionPool

# 导入认证相关模块
from auth import create_users_table, set_global_pool, get_current_admin_user, get_current_user, get_current_contributor_user, UserResponse, UserRole
from fastapi import Request
from auth_routes import auth_router, set_user_manager as set_auth_user_manager, init_ustc_oauth
from auth_middleware import create_auth_middleware
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import VectorParams, Distance, FieldCondition, MatchValue
from oss2 import Auth, Bucket
from starlette.responses import StreamingResponse
from chunks2embedding import (
    embedding_init,
    upsert_md_file,
    upsert_md_file_with_source,
    upsert_md_file_with_original,
    upsert_qa_pair,
    delete_by_source,
    list_all_collections,
    get_collection_info
)

# 尝试导入marker转换器，如果失败则使用轻量级转换器
try:
    from marker_pdf_converter import convert_pdf_to_markdown_with_marker
    MARKER_AVAILABLE = True
    print("✅ Marker转换器可用")
except ImportError as e:
    MARKER_AVAILABLE = False
    print(f"⚠️ Marker转换器不可用: {str(e)}，将使用轻量级转换器")

from pdf import (
    register_user_document_tool,
    get_user_document_tool,
    get_user_document_tool_by_session,
    list_user_document_tools,
    set_db_pool,
    load_user_document_tools_from_db
)
from docx import Document
from smart_search import create_search_tool
import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', "app.log"), encoding='utf-8')
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

# 只输出到文件，不输出到控制台（避免日志过多）
# chat_console_handler = logging.StreamHandler()
# chat_console_handler.setFormatter(logging.Formatter("%(asctime)s [CHAT] %(message)s"))
# chat_logger.addHandler(chat_console_handler)

# 创建统一的日志记录器
logger = logging.getLogger("unified-service")
logger.setLevel(logging.INFO)
# 防止日志传播到父级logger，避免重复输出
logger.propagate = False

# 创建统一服务日志文件处理器
unified_file_handler = logging.FileHandler(os.path.join(log_dir, "unified_service.log"), encoding='utf-8')
unified_file_handler.setFormatter(logging.Formatter("%(asctime)s [SERVICE] %(message)s"))
logger.addHandler(unified_file_handler)

# 同时输出到控制台（实时看到服务日志）
unified_console_handler = logging.StreamHandler()
unified_console_handler.setFormatter(logging.Formatter("%(asctime)s [SERVICE] %(message)s"))
logger.addHandler(unified_console_handler)

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
        logger.info(f"当前显存状态: {self.get_gpu_memory_info()}")
        
        await self.lock.acquire()
        
        try:
            # 如果已经有其他模型加载，先清理
            if self.current_model and self.current_model != model_type:
                logger.info(f"检测到不同模型类型，先清理 {self.current_model} 模型...")
                await self.release()
            
            # 检查显存是否足够
            required_memory = 2048 if model_type == "ocr" else 512  # OCR需要更多显存，embedding现在只需要512MB
            if not self.check_gpu_memory_available(required_memory):
                logger.warning(f"GPU显存不足，尝试强制清理...")
                self.clear_gpu_memory()
                # 再次检查
                if not self.check_gpu_memory_available(required_memory):
                    raise RuntimeError(f"GPU显存不足，需要至少 {required_memory}MB 显存")
            
            # 记录当前使用的模型类型
            self.current_model = model_type
            logger.info(f"已获取GPU资源，准备使用 {model_type} 模型")
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
        
    def get_gpu_memory_info(self):
        """获取GPU内存信息"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            cached = torch.cuda.memory_reserved() / 1024**2  # MB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            return f"已分配: {allocated:.2f}MB, 已缓存: {cached:.2f}MB, 总计: {total:.2f}MB"
        else:
            return "GPU不可用"
    
    def check_gpu_memory_available(self, required_mb):
        """检查GPU是否有足够的内存"""
        if not torch.cuda.is_available():
            return False
        
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
        available = total - allocated
        
        return available >= required_mb
    
    def clear_gpu_memory(self):
        """清理GPU内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info(f"GPU内存已清理 - 已释放 {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
def get_ollama_model(self):
        """获取DeepSeek API模型实例（按需加载）"""
        if "ollama" not in self.model_instances:
            # 使用DeepSeek API
            # =============== DeepSeek API 配置 ===============
            # 获取API Key: https://platform.deepseek.com/
            # 支持的模型: deepseek-chat, deepseek-coder, deepseek-r3
            api_key = os.getenv("DEEPSEEK_API_KEY", "")  # DeepSeek API Key（必须从环境变量设置）
            api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")  # DeepSeek API 地址
            model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # DeepSeek 模型名称
            # ================================================
            
            self.model_instances["ollama"] = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=api_base,
                max_tokens=10000,  # API模型可以处理更多token
                temperature=0.1,
                request_timeout=120.0,  # 增加超时时间
                max_retries=5,  # 增加重试次数
                streaming=False  # 禁用流式响应避免卡住
            )
            logger.info(f"🚀 使用DeepSeek API模型: {model_name}")
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
            from marker_pdf_converter import convert_pdf_to_markdown_with_marker
            self.model_instances["ocr"] = convert_pdf_to_markdown_with_marker
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

def get_current_user_from_token(request):
    """从请求中获取当前用户信息"""
    try:
        # 检查request是否有headers属性（FastAPI Request对象）
        if hasattr(request, 'headers'):
            # 从请求头中获取Authorization token (不区分大小写)
            auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
            logger.info(f"🔍 调试认证: auth_header = {auth_header}")
            
            if not auth_header or not auth_header.startswith("Bearer "):
                logger.info("❌ 没有找到有效的Authorization头")
                return None
            
            token = auth_header.split(" ")[1]
            logger.info(f"🔍 提取的token: {token[:20]}...")
            
            # 验证token并获取用户信息
            from auth import verify_token
            token_data = verify_token(token)
            logger.info(f"🔍 token验证结果: {token_data}")
            
            # 暂时允许测试token
            if not token_data and token == "test":
                token_data = {"username": "test_user"}
                logger.info("🔍 使用测试token")
            
            if not token_data:
                logger.info("❌ token验证失败")
                return None
            
            # 这里需要根据实际的用户管理逻辑来获取用户信息
            # 暂时返回一个简单的用户对象
            class SimpleUser:
                def __init__(self, username):
                    self.id = username  # 使用用户名作为ID
                    self.username = username
            
            username = token_data.get("username") if isinstance(token_data, dict) else token_data.username
            user = SimpleUser(username)
            logger.info(f"✅ 成功获取用户: {user.username}")
            return user
        else:
            # 如果是ChatRequest对象，暂时返回None（需要从其他地方获取认证信息）
            logger.info("❌ 请求对象没有headers属性，无法获取认证信息")
            return None
    except Exception as e:
        logger.error(f"获取用户信息失败: {str(e)}", exc_info=True)
        return None

# OSS 配置（从环境变量读取）
OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET", "")
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "https://oss-cn-hangzhou.aliyuncs.com")
OSS_BUCKET = os.getenv("OSS_BUCKET", "")

# 本地临时路径
LOCAL_DIR = "/home/user/ustcchat/oss"
os.makedirs(LOCAL_DIR, exist_ok=True)

# ======================
# 创建路由
# ======================
kb_router = APIRouter(prefix="/kb", tags=["知识库管理"])
agent_router = APIRouter(prefix="/agent", tags=["对话Agent"])

# ======================
# 知识库名称映射配置
# ======================
KNOWLEDGE_BASE_NAME_MAPPING = {
    "nsrl_tech_docs": "NSRL技术文档库",
    "test": "测试知识库",
    "default": "默认知识库"
}

def get_display_name(technical_name: str) -> str:
    """获取知识库的显示名称"""
    return KNOWLEDGE_BASE_NAME_MAPPING.get(technical_name, technical_name)

def get_technical_name(display_name: str) -> str:
    """根据显示名称获取技术名称"""
    for tech_name, disp_name in KNOWLEDGE_BASE_NAME_MAPPING.items():
        if disp_name == display_name:
            return tech_name
    return display_name

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
            "display_name": get_display_name(kb_name),
            "exists": True,
            "points_count": len(all_points),
            "documents": list(document_names),
            "document_count": len(document_names)
        }
    except Exception as e:
        logger.error(f"获取知识库信息失败: {str(e)}", exc_info=True)
        return {
            "name": kb_name,
            "display_name": get_display_name(kb_name),
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

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """安全响应头中间件，防止HTTP响应头注入攻击"""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # 清理所有响应头，移除可能的CRLF注入字符
        cleaned_headers = {}
        for key, value in response.headers.items():
            # 清理键名
            clean_key = re.sub(r'[\r\n]', '', key)
            # 清理值
            clean_value = re.sub(r'[\r\n]', '', str(value))
            cleaned_headers[clean_key] = clean_value
        
        # 重新设置响应头
        # 清除现有头部
        for key in list(response.headers.keys()):
            del response.headers[key]
        # 设置清理后的头部
        for key, value in cleaned_headers.items():
            response.headers[key] = value
            
        return response

def sanitize_filename(filename: str) -> str:
    """清理文件名，移除所有可能导致HTTP响应头注入的字符"""
    if not filename:
        return f"uploaded_file_{int(time.time())}"
    
    # 移除控制字符（包括\r、\n、\t等）
    safe_filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
    # 移除可能导致路径遍历的字符
    safe_filename = safe_filename.replace('..', '_')
    safe_filename = safe_filename.replace('/', '_')
    safe_filename = safe_filename.replace('\\', '_')
    # 移除其他特殊字符
    safe_filename = re.sub(r'[<>:"|?*]', '_', safe_filename)
    # 移除多余的空格和点
    safe_filename = safe_filename.strip('. ')
    # 确保文件名不为空
    if not safe_filename.strip():
        safe_filename = f"uploaded_file_{int(time.time())}"
    
    return safe_filename

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


def process_uploaded_file(file: UploadFile, knowledge_base: str) -> Dict[str, Any]:
    """处理上传的文件并添加到知识库"""
    try:
        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        file_extension = os.path.splitext(file.filename)[1].lower()
        # 清理文件名，移除所有可能导致HTTP响应头注入的字符
        safe_filename = sanitize_filename(file.filename)
        temp_file_path = os.path.join(temp_dir, safe_filename)
        
        # 保存上传的文件
        with open(temp_file_path, "wb") as buffer:
            content = file.file.read()
            buffer.write(content)
        
        logger.info(f"文件已保存到临时路径: {temp_file_path}")
        
        # 根据文件类型处理
        if file_extension in ['.pdf', '.docx', '.ppt', '.pptx', '.xls', '.xlsx']:
            # 其他格式优先使用marker转换器，轻量级转换器作为备用
            try:
                logger.info(f"🔄 使用marker转换器处理文档: {file.filename}")
                return process_document_with_marker(temp_file_path, knowledge_base, file.filename)
            except Exception as marker_error:
                logger.warning(f"⚠️ marker转换失败，尝试轻量级转换器: {str(marker_error)}")
                try:
                    logger.info(f"🔄 尝试使用轻量级转换器处理文档: {file.filename}")
                    return process_document_file(temp_file_path, knowledge_base, file.filename)
                except Exception as lightweight_error:
                    logger.error(f"❌ 轻量级转换也失败: {str(lightweight_error)}")
                    return {
                        "success": False,
                        "message": f"文档转换失败。Marker错误: {str(marker_error)}。轻量级转换器错误: {str(lightweight_error)}",
                        "data": {"filename": file.filename}
                    }
        elif file_extension in ['.md', '.markdown']:
            return process_markdown_file(temp_file_path, knowledge_base, file.filename)
        elif file_extension == '.txt':
            return process_text_file(temp_file_path, knowledge_base, file.filename)
        else:
            return {
                "success": False,
                "message": f"不支持的文件格式: {file_extension}。支持的格式: PDF, Word(.docx), PowerPoint, Excel, Markdown, TXT",
                "data": {"filename": file.filename}
            }
    
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}")
        return {
            "success": False,
            "message": f"处理文件时出错: {str(e)}",
            "data": {"filename": file.filename}
        }
    finally:
        # 清理临时文件
        try:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"清理临时文件失败: {str(e)}")


def process_document_with_marker(file_path: str, knowledge_base: str, filename: str) -> Dict[str, Any]:
    """使用独立marker进程处理多种文档格式（PDF、Word、PowerPoint、Excel）"""
    try:
        # 获取文件名（去掉路径和扩展名）
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 设置输出目录
        output_dir = f'/home/user/ustcchat/ustc/marker_outputs/{base_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用独立marker进程转换
        logger.info(f"🔄 使用独立marker进程转换文档: {file_path}")
        
        # 直接调用独立marker脚本，避免导入问题
        import subprocess
        import json
        
        result = subprocess.run(
            ['/home/user/miniconda3/envs/langchain/bin/python', 'marker_standalone.py', file_path, output_dir, base_name],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if result.returncode != 0:
            raise Exception(f"独立marker进程执行失败: {result.stderr}")
        
        # 解析结果
        try:
            result_data = json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            # 如果没有JSON输出，检查是否生成了文件
            expected_md_file = os.path.join(output_dir, f"{base_name}.md")
            if os.path.exists(expected_md_file):
                with open(expected_md_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                result_data = {
                    "success": True,
                    "text_length": len(text),
                    "md_file": expected_md_file,
                    "method": "marker_standalone"
                }
            else:
                raise Exception(f"独立marker进程执行失败，无JSON输出且未生成文件")
        
        if not result_data.get("success"):
            raise Exception(f"Marker转换失败: {result_data.get('error', '未知错误')}")
        
        # 生成的markdown文件路径
        md_file_path = result_data.get("md_file", os.path.join(output_dir, f"{base_name}.md"))
        
        if not os.path.exists(md_file_path):
            raise Exception(f"marker转换失败，未生成markdown文件。期望路径: {md_file_path}")
        
        # 添加到知识库 - 使用新函数存储原文件内容
        vector_store = embedding_init(collection_name=knowledge_base)
        operation_info = upsert_md_file_with_original(md_file_path, vector_store)
        
        return {
            "success": True,
            "message": f"文档文件 {filename} 处理成功（marker独立进程）",
            "data": {
                "filename": filename,
                "operation_info": operation_info,
                "converter_result": {
                    "file_path": file_path,
                    "output_dir": output_dir,
                    "text_length": result_data.get("text_length", 0),
                    "method": "marker_standalone"
                },
                "method": "marker_standalone"
            }
        }
        
    except Exception as e:
        logger.error(f"marker独立进程处理文档文件失败: {str(e)}")
        return {
            "success": False,
            "message": f"marker独立进程处理文档文件失败: {str(e)}",
            "data": {"filename": filename}
        }


def process_document_file(file_path: str, knowledge_base: str, filename: str) -> Dict[str, Any]:
    """处理多种文档格式（PDF、Word、PowerPoint、Excel）"""
    try:
        # 获取文件名（去掉路径和扩展名）
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 设置输出目录
        output_dir = f'/home/user/ustcchat/ustc/marker_outputs/{base_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用轻量级多格式转换器
        logger.info(f"🔄 使用轻量级转换器处理文档: {file_path}")
        from lightweight_marker_converter import convert_with_lightweight_marker
        result = convert_with_lightweight_marker(
            file_path=file_path,
            output_dir=output_dir,
            base_name=base_name
        )
        
        if not result["success"]:
            return {
                "success": False,
                "message": f"文档转换失败: {result['message']}",
                "data": {"filename": filename}
            }
        
        # 生成的markdown文件路径
        md_file_path = os.path.join(output_dir, f"{base_name}.md")
        
        if not os.path.exists(md_file_path):
            return {
                "success": False,
                "message": f"文档转换失败，未生成markdown文件。期望路径: {md_file_path}",
                "data": {"filename": filename}
            }
        
        # 添加到知识库 - 使用新函数存储原文件内容
        vector_store = embedding_init(collection_name=knowledge_base)
        operation_info = upsert_md_file_with_original(md_file_path, vector_store)
        
        return {
            "success": True,
            "message": f"文档文件 {filename} 处理成功",
            "data": {
                "filename": filename,
                "operation_info": operation_info,
                "converter_result": result["data"]
            }
        }
    
    except Exception as e:
        logger.error(f"处理文档文件失败: {str(e)}")
        return {
            "success": False,
            "message": f"处理文档文件失败: {str(e)}",
            "data": {"filename": filename}
        }


def process_pdf_file(file_path: str, knowledge_base: str, filename: str) -> Dict[str, Any]:
    """处理PDF文件"""
    try:
        # 获取文件名（去掉路径和扩展名）
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 设置输出目录
        output_dir = f'/home/user/ustcchat/ustc/marker_outputs/{base_name}'
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用轻量级多格式转换器
        logger.info(f"🔄 使用轻量级转换器处理文件: {file_path}")
        from lightweight_marker_converter import convert_with_lightweight_marker
        result = convert_with_lightweight_marker(
            file_path=file_path,
            output_dir=output_dir,
            base_name=base_name
        )
        
        if not result["success"]:
            return {
                "success": False,
                "message": f"PDF转换失败: {result['message']}",
                "data": {"filename": filename}
            }
        
        # 生成的markdown文件路径
        md_file_path = os.path.join(output_dir, f"{base_name}.md")
        
        if not os.path.exists(md_file_path):
            return {
                "success": False,
                "message": f"PDF转换失败，未生成markdown文件。期望路径: {md_file_path}",
                "data": {"filename": filename}
            }
        
        # 添加到知识库 - 使用新函数存储原文件内容
        vector_store = embedding_init(collection_name=knowledge_base)
        operation_info = upsert_md_file_with_original(md_file_path, vector_store)
        
        return {
            "success": True,
            "message": f"PDF文件 {filename} 处理成功",
            "data": {
                "filename": filename,
                "operation_info": operation_info,
                "marker_result": result["data"]
            }
        }
    
    except Exception as e:
        logger.error(f"处理PDF文件失败: {str(e)}")
        return {
            "success": False,
            "message": f"处理PDF文件失败: {str(e)}",
            "data": {"filename": filename}
        }


def process_markdown_file(file_path: str, knowledge_base: str, filename: str) -> Dict[str, Any]:
    """处理Markdown文件"""
    try:
        # 直接使用markdown文件
        vector_store = embedding_init(collection_name=knowledge_base)
        operation_info = upsert_md_file_with_original(file_path, vector_store)
        
        return {
            "success": True,
            "message": f"Markdown文件 {filename} 处理成功",
            "data": {
                "filename": filename,
                "operation_info": operation_info
            }
        }
    
    except Exception as e:
        logger.error(f"处理Markdown文件失败: {str(e)}")
        return {
            "success": False,
            "message": f"处理Markdown文件失败: {str(e)}",
            "data": {"filename": filename}
        }


def process_word_file(file_path: str, knowledge_base: str, filename: str) -> Dict[str, Any]:
    """处理Word文档"""
    try:
        # 读取Word文档内容
        doc = Document(file_path)
        text_content = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_content.append(paragraph.text.strip())
        
        # 将内容转换为markdown格式
        content = "\n\n".join(text_content)
        
        # 创建临时markdown文件
        temp_md_path = file_path.replace('.docx', '.md').replace('.doc', '.md')
        with open(temp_md_path, 'w', encoding='utf-8') as f:
            f.write(f"# {os.path.splitext(filename)[0]}\n\n{content}")
        
        # 添加到知识库 - 使用新函数存储原文件内容
        vector_store = embedding_init(collection_name=knowledge_base)
        operation_info = upsert_md_file_with_original(temp_md_path, vector_store)
        
        # 清理临时文件
        os.remove(temp_md_path)
        
        return {
            "success": True,
            "message": f"Word文档 {filename} 处理成功",
            "data": {
                "filename": filename,
                "operation_info": operation_info
            }
        }
    
    except Exception as e:
        logger.error(f"处理Word文档失败: {str(e)}")
        return {
            "success": False,
            "message": f"处理Word文档失败: {str(e)}",
            "data": {"filename": filename}
        }


def process_text_file(file_path: str, knowledge_base: str, filename: str) -> Dict[str, Any]:
    """处理纯文本文件"""
    try:
        # 读取文本内容
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # 创建临时markdown文件，但使用原始文件名
        temp_md_path = file_path.replace('.txt', '.md')
        with open(temp_md_path, 'w', encoding='utf-8') as f:
            f.write(f"# {os.path.splitext(filename)[0]}\n\n{content}")
        
        # 添加到知识库，传递原始文件名
        vector_store = embedding_init(collection_name=knowledge_base)
        operation_info = upsert_md_file_with_source(temp_md_path, vector_store, filename)
        
        # 清理临时文件
        os.remove(temp_md_path)
        
        return {
            "success": True,
            "message": f"文本文件 {filename} 处理成功",
            "data": {
                "filename": filename,
                "operation_info": operation_info
            }
        }
    
    except Exception as e:
        logger.error(f"处理文本文件失败: {str(e)}", exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"TXT文件处理详细错误: {error_details}")
        return {
            "success": False,
            "message": f"处理文本文件失败: {str(e)}",
            "data": {
                "filename": filename,
                "error_details": error_details
            }
        }


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
    filename: Optional[str] = None  # 新增：用于重命名下载的文件

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

# 新增：知识库查询请求模型
class KnowledgeBaseQueryRequest(BaseModel):
    knowledge_base_name: str
    query: str
    search_type: str = "hybrid"  # "vector", "keyword", "hybrid"
    top_k: Optional[int] = 10
    similarity_threshold: Optional[float] = 0.5
    keyword_match_threshold: Optional[int] = 1

# 新增：知识库查询结果模型
class QueryResult(BaseModel):
    content: str
    document_name: str
    title: str
    score: float
    search_type: str
    metadata: Dict[str, Any]
    is_qa_pair: Optional[bool] = None  # 是否为问答对
    question: Optional[str] = None  # 如果是QA对，分离的问题
    answer: Optional[str] = None  # 如果是QA对，分离的答案

def query_knowledge_base_sync(knowledge_base_name: str, query: str, search_type: str = "hybrid",
                            top_k: int = 20, similarity_threshold: float = 0.3,
                            keyword_match_threshold: int = 1) -> List[QueryResult]:
    """同步查询知识库内容"""
    try:
        global qdrant_client
        if qdrant_client is None:
            qdrant_client = QdrantClient(host="localhost", port=6333)

        # 检查知识库是否存在
        try:
            qdrant_client.get_collection(knowledge_base_name)
        except:
            logger.error(f"知识库 '{knowledge_base_name}' 不存在")
            return []

        # 使用向量化模型进行查询
        from chunks2embedding import embedding_init
        vector_store = embedding_init(collection_name=knowledge_base_name)
        search_results = vector_store.weighted_hybrid_search(query=query, k=top_k)

        results = []
        for doc, score in search_results:
            if score < similarity_threshold:
                continue

            # 获取文档名
            document_name = doc.metadata.get("source", "未知文档")
            if document_name.endswith('.md'):
                document_name = document_name[:-3]

            # 检查是否为问答对
            is_qa_pair = doc.metadata.get("is_qa_pair", False) or doc.metadata.get("type") == "qa"
            question = None
            answer = None

            if is_qa_pair:
                # 从metadata中提取问题和答案
                question = doc.metadata.get("question", "")
                answer = doc.metadata.get("answer", "")
                # 或者从内容中解析
                if not question and "问题：" in doc.page_content:
                    try:
                        parts = doc.page_content.split("\n\n", 1)
                        if len(parts) >= 1:
                            question_line = parts[0]
                            if question_line.startswith("问题："):
                                question = question_line[3:].strip()
                        if len(parts) >= 2:
                            answer_part = parts[1]
                            if answer_part.startswith("答案："):
                                answer = answer_part[3:].strip()
                    except:
                        pass

            result = QueryResult(
                content=doc.page_content,
                document_name=document_name,
                title=doc.metadata.get("title", "无标题"),
                score=score,
                search_type="hybrid",
                metadata=doc.metadata,
                is_qa_pair=is_qa_pair,
                question=question,
                answer=answer
            )
            results.append(result)

        logger.info(f"知识库查询完成，找到 {len(results)} 个结果")
        return results

    except Exception as e:
        logger.error(f"查询知识库失败: {str(e)}")
        return []

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
                            "title": VectorParams(size=896, distance=Distance.COSINE),
                            "content": VectorParams(size=896, distance=Distance.COSINE)
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
                
                # 如果提供了filename参数，重命名文件
                if request.filename:
                    # 确保filename有.pdf扩展名
                    if not request.filename.endswith('.pdf'):
                        request.filename = f"{request.filename}.pdf"
                    
                    # 生成新的文件路径
                    renamed_file_path = os.path.join(LOCAL_DIR, request.filename)
                    
                    # 重命名文件
                    try:
                        os.rename(local_input, renamed_file_path)
                        local_input = renamed_file_path
                        logger.info(f"文件已重命名: {orginal_file_name} -> {request.filename}")
                        chat_logger.info(f"📁 文件重命名: {orginal_file_name} -> {request.filename}")
                    except Exception as e:
                        logger.error(f"文件重命名失败: {str(e)}")
                        chat_logger.error(f"❌ 文件重命名失败: {str(e)}")
                        # 重命名失败不影响后续处理，继续使用原文件名
                
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
                
                # 确定用于生成md文件路径的文件名
                # 如果提供了filename，使用filename（去掉.pdf扩展名）
                # 否则使用document_name
                if request.filename:
                    # 去掉.pdf扩展名
                    base_filename = request.filename.replace('.pdf', '')
                    tempfile = f'/home/user/ustcchat/ustc/marker_outputs/{base_filename}/{base_filename}.md'
                    logger.info(f"使用重命名后的文件名生成md路径: {tempfile}")
                    chat_logger.info(f"📄 使用重命名后的文件名生成md路径: {tempfile}")
                else:
                    tempfile = f'/home/user/ustcchat/ustc/marker_outputs/{request.document_name}/{request.document_name}.md'
                    logger.info(f"使用原始文件名生成md路径: {tempfile}")
                    chat_logger.info(f"📄 使用原始文件名生成md路径: {tempfile}")
                
                # =============== 显存优化：使用向量化模型 ===============
                logger.info("开始使用向量化模型处理文档...")
                # 直接使用embedding模型，不需要GPU资源管理
                from chunks2embedding import embedding_init
                vector_store = embedding_init(collection_name=request.name)
                operation_info = upsert_md_file(tempfile, vector_store)
                logger.info("向量化模型处理完成")
                
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
            # 直接使用embedding模型，不需要GPU资源管理
            from chunks2embedding import embedding_init
            vector_store = embedding_init(collection_name=request.name)
            operation_info = delete_by_source(deletename, vector_store)
            logger.info("向量化模型操作完成")
            
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
                "display_name": kb_info["display_name"],
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

# ======================
# 问答对上传API（统一接口）
# ======================
# 说明：
# 1. 单个上传：POST /api/qa-pair，传入单个QAPairRequest对象
# 2. 批量上传：POST /api/qa-pairs/batch，传入QAPairRequest对象列表
# 3. 两个接口内部都调用同一个处理逻辑，实现代码复用
# 4. 单个上传实际上是批量上传的特例（列表长度为1）
# ======================

# 删除重复的单个上传API端点，使用统一的批量接口

# 新增：批量问答对上传API端点（兼容单个）
@kb_router.post("/api/qa-pairs/batch", response_model=KnowledgeBaseResponse)
async def upload_qa_pairs_batch(request: List[QAPairRequest]):
    """批量上传问答对到指定知识库（兼容单个上传）"""
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
        # 直接使用embedding模型，不需要GPU资源管理
        from chunks2embedding import embedding_init
        vector_store = embedding_init(collection_name=knowledge_base_name)
        
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
        
        logger.info("向量化模型批量处理完成")
        
        # 获取更新后的知识库信息
        current_kb = get_current_knowledge_base_info(knowledge_base_name)
        
        # 根据上传数量调整返回消息
        if len(request) == 1:
            message = f"问答对已成功添加到知识库 '{knowledge_base_name}'"
        else:
            message = f"批量上传完成：成功 {success_count} 个，失败 {failed_count} 个"
        
        return KnowledgeBaseResponse(
            success=True,
            message=message,
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

# 新增：统一问答对上传API端点（推荐使用）
@kb_router.post("/api/qa-pair", response_model=KnowledgeBaseResponse)
async def upload_qa_pair_unified(request: QAPairRequest):
    """统一问答对上传API端点（内部调用批量接口）"""
    try:
        # 将单个请求包装成列表，调用批量接口
        batch_request = [request]
        return await upload_qa_pairs_batch(batch_request)
        
    except Exception as e:
        logger.error(f"统一问答对上传失败: {str(e)}", exc_info=True)
        return KnowledgeBaseResponse(
            success=False,
            message=f"统一问答对上传失败: {str(e)}",
            data={"name": request.knowledge_base_name}
        )

# 新增：Markdown文件上传请求模型
class MarkdownFileRequest(BaseModel):
    knowledge_base_name: str
    file_name: str
    file_path: str  # 直接使用文件路径，而不是文件内容

# 新增：批量Markdown文件上传API端点
@kb_router.post("/api/md-files/batch", response_model=KnowledgeBaseResponse)
async def upload_md_files_batch(request: List[MarkdownFileRequest]):
    """批量上传Markdown文件到指定知识库（兼容单个上传）"""
    try:
        global qdrant_client
        if not request:
            return KnowledgeBaseResponse(
                success=False,
                message="请求列表不能为空",
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
                data={}
            )
        
        # 使用向量化模型处理文档
        # 直接使用embedding模型，不需要GPU资源管理
        from chunks2embedding import embedding_init
        vector_store = embedding_init(collection_name=knowledge_base_name)
        
        uploaded_files = []
        for md_request in request:
            try:
                # 确保文件名以.md结尾
                filename = md_request.file_name
                if not filename.endswith('.md'):
                    filename = f"{filename}.md"
                
                # 检查文件路径是否存在
                if not os.path.exists(md_request.file_path):
                    raise FileNotFoundError(f"文件不存在: {md_request.file_path}")
                
                # 直接使用文件路径上传到向量数据库
                operation_info = upsert_md_file(md_request.file_path, vector_store)
                
                uploaded_files.append({
                    "file_name": filename,
                    "file_path": md_request.file_path,
                    "operation_info": str(operation_info)
                })
                
            except Exception as e:
                logger.error(f"上传文件 {md_request.file_name} 失败: {str(e)}")
                uploaded_files.append({
                    "file_name": md_request.file_name,
                    "file_path": md_request.file_path,
                    "error": str(e)
                })
        
        return KnowledgeBaseResponse(
            success=True,
            message=f"成功上传 {len(uploaded_files)} 个Markdown文件到知识库 '{knowledge_base_name}'",
            data={
                "knowledge_base_name": knowledge_base_name,
                "uploaded_files": uploaded_files,
                "total_files": len(request)
            }
        )
        
    except Exception as e:
        logger.error(f"批量Markdown文件上传失败: {str(e)}", exc_info=True)
        return KnowledgeBaseResponse(
            success=False,
            message=f"批量Markdown文件上传失败: {str(e)}",
            data={}
        )

# 新增：单个Markdown文件上传API端点
@kb_router.post("/api/md-file", response_model=KnowledgeBaseResponse)
async def upload_md_file_unified(request: MarkdownFileRequest):
    """统一Markdown文件上传API端点（内部调用批量接口）"""
    try:
        # 将单个请求包装成列表，调用批量接口
        batch_request = [request]
        return await upload_md_files_batch(batch_request)
        
    except Exception as e:
        logger.error(f"Markdown文件上传失败: {str(e)}", exc_info=True)
        return KnowledgeBaseResponse(
            success=False,
            message=f"Markdown文件上传失败: {str(e)}",
            data={}
        )

# ======================
# 文件上传API
# ======================
@kb_router.post("/api/upload-file")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    knowledge_base: str = Form(...),
    current_user: UserResponse = Depends(get_current_contributor_user)
):
    """上传文件到知识库（支持PDF、MD、Word、TXT等格式）"""
    try:
        # 清理文件名，防止HTTP响应头注入攻击
        original_filename = file.filename
        safe_filename = sanitize_filename(file.filename)
        
        # 记录详细的请求信息用于调试
        logger.info(f"文件上传请求详情:")
        logger.info(f"  - 原始文件名: {original_filename}")
        logger.info(f"  - 清理后文件名: {safe_filename}")
        logger.info(f"  - 文件大小: {file.size if hasattr(file, 'size') else 'unknown'}")
        logger.info(f"  - 文件类型: {file.content_type}")
        logger.info(f"  - 知识库: {knowledge_base}")
        logger.info(f"  - 请求URL: {request.url}")
        logger.info(f"  - 请求头: {dict(request.headers)}")
        logger.info(f"  - 客户端IP: {request.client.host if request.client else 'unknown'}")
        
        # 验证文件类型（使用原始文件名进行扩展名检查）
        allowed_extensions = ['.pdf', '.md', '.markdown', '.docx', '.txt']
        file_extension = os.path.splitext(original_filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            return {
                "success": False,
                "message": f"不支持的文件格式: {file_extension}。支持的格式: {', '.join(allowed_extensions)}",
                "data": {"filename": safe_filename}
            }
        
        # 验证知识库是否存在
        try:
            collections = qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            if knowledge_base not in collection_names:
                return {
                    "success": False,
                    "message": f"知识库 '{knowledge_base}' 不存在",
                    "data": {"filename": safe_filename}
                }
        except Exception as e:
            logger.error(f"验证知识库失败: {str(e)}")
            return {
                "success": False,
                "message": f"验证知识库失败: {str(e)}",
                "data": {"filename": safe_filename}
            }
        
        # 处理文件
        result = process_uploaded_file(file, knowledge_base)
        
        if result["success"]:
            logger.info(f"文件 {safe_filename} 上传成功")
        else:
            logger.error(f"文件 {safe_filename} 上传失败: {result['message']}")
        
        return result
        
    except Exception as e:
        logger.error(f"文件上传API出错: {str(e)}", exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"详细错误信息: {error_details}")
        return {
            "success": False,
            "message": f"文件上传失败: {str(e)}",
            "data": {
                "filename": safe_filename if 'safe_filename' in locals() else "unknown",
                "error_details": error_details
            }
        }


# ======================
# 删除API（统一接口）
# ======================
# 说明：
# 1. 单个删除：POST /api/delete，传入单个DeleteRequest对象
# 2. 批量删除：POST /api/delete/batch，传入DeleteRequest对象列表
# 3. 两个接口内部都调用同一个处理逻辑，实现代码复用
# 4. 支持删除文档和问答对
# ======================

# 删除请求模型
class DeleteRequest(BaseModel):
    knowledge_base_name: str
    document_name: str  # 要删除的文档名或问答对名
    delete_type: str = "document"  # "document" 或 "qa_pair"，默认为文档

# 统一删除API端点（兼容单个和批量）
@kb_router.post("/api/delete/batch", response_model=KnowledgeBaseResponse)
async def delete_items_batch(request: List[DeleteRequest]):
    """批量删除文档或问答对（兼容单个删除）"""
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
        logger.info(f"开始使用向量化模型批量删除 {len(request)} 个项目...")
        # 直接使用embedding模型，不需要GPU资源管理
        from chunks2embedding import embedding_init
        vector_store = embedding_init(collection_name=knowledge_base_name)
        
        success_count = 0
        failed_count = 0
        failed_items = []
        deleted_items = []
        
        for delete_request in request:
            try:
                # 根据删除类型处理
                if delete_request.delete_type == "qa_pair":
                    # 删除问答对：直接使用文档名作为source
                    document_name = delete_request.document_name
                    if not document_name.endswith('.md'):
                        document_name = f"{document_name}.md"
                    
                    operation_info = delete_by_source(document_name, vector_store)
                    deleted_items.append({
                        "document_name": document_name,
                        "type": "qa_pair",
                        "operation_info": str(operation_info)
                    })
                else:
                    # 删除文档：直接使用文档名作为source
                    document_name = delete_request.document_name
                    # 不再自动添加.md后缀，因为新文件使用原始文件名作为source
                    
                    operation_info = delete_by_source(document_name, vector_store)
                    deleted_items.append({
                        "document_name": document_name,
                        "type": "document",
                        "operation_info": str(operation_info)
                    })
                
                success_count += 1
                logger.info(f"成功删除: {document_name}")
                
            except Exception as e:
                failed_count += 1
                failed_items.append({
                    "document_name": delete_request.document_name,
                    "type": delete_request.delete_type,
                    "error": str(e)
                })
                logger.error(f"删除失败: {delete_request.document_name}, 错误: {str(e)}")
        
        logger.info("向量化模型批量删除完成")
        
        # 获取更新后的知识库信息
        current_kb = get_current_knowledge_base_info(knowledge_base_name)
        
        # 根据删除数量调整返回消息
        if len(request) == 1:
            if request[0].delete_type == "qa_pair":
                message = f"问答对 '{request[0].document_name}' 已从知识库 '{knowledge_base_name}' 中删除"
            else:
                message = f"文档 '{request[0].document_name}' 已从知识库 '{knowledge_base_name}' 中删除"
        else:
            message = f"批量删除完成：成功 {success_count} 个，失败 {failed_count} 个"
        
        return KnowledgeBaseResponse(
            success=True,
            message=message,
            data={
                "name": knowledge_base_name,
                "total_requested": len(request),
                "success_count": success_count,
                "failed_count": failed_count,
                "failed_items": failed_items if failed_items else None,
                "deleted_items": deleted_items if deleted_items else None,
                "document_count": current_kb["document_count"],
                "points_count": current_kb["points_count"],
                "documents": current_kb["documents"]
            }
        )
        
    except Exception as e:
        logger.error(f"批量删除失败: {str(e)}", exc_info=True)
        return KnowledgeBaseResponse(
            success=False,
            message=f"批量删除失败: {str(e)}",
            data={}
        )

# 单个删除API端点（推荐使用）
@kb_router.post("/api/delete", response_model=KnowledgeBaseResponse)
async def delete_item_unified(request: DeleteRequest):
    """统一删除API端点（内部调用批量接口）"""
    try:
        # 将单个请求包装成列表，调用批量接口
        batch_request = [request]
        return await delete_items_batch(batch_request)
        
    except Exception as e:
        logger.error(f"统一删除失败: {str(e)}", exc_info=True)
        return KnowledgeBaseResponse(
            success=False,
            message=f"统一删除失败: {str(e)}",
            data={"name": request.knowledge_base_name}
        )

# 新增：知识库查询API端点
@kb_router.post("/api/query", response_model=KnowledgeBaseResponse)
async def query_knowledge_base(request: KnowledgeBaseQueryRequest):
    """查询知识库内容，支持向量搜索、关键词搜索和混合搜索"""
    try:
        logger.info(f"收到知识库查询请求: {request.knowledge_base_name}, 查询: {request.query}")

        # 执行查询
        results = query_knowledge_base_sync(
            knowledge_base_name=request.knowledge_base_name,
            query=request.query,
            search_type=request.search_type,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            keyword_match_threshold=request.keyword_match_threshold
        )

        if not results:
            return KnowledgeBaseResponse(
                success=True,
                message=f"未找到相关结果，尝试调整查询条件或检查知识库内容",
                data={
                    "query": request.query,
                    "knowledge_base_name": request.knowledge_base_name,
                    "search_type": request.search_type,
                    "results": [],
                    "total_results": 0
                }
            )

        # 转换为字典格式
        results_data = []
        for result in results:
            result_dict = {
                "content": result.content,
                "document_name": result.document_name,
                "title": result.title,
                "score": result.score,
                "search_type": result.search_type,
                "metadata": result.metadata,
                "is_qa_pair": result.is_qa_pair
            }

            # 如果是问答对，添加分离的问题和答案
            if result.is_qa_pair and result.question and result.answer:
                result_dict["question"] = result.question
                result_dict["answer"] = result.answer

            results_data.append(result_dict)

        return KnowledgeBaseResponse(
            success=True,
            message=f"查询完成，找到 {len(results)} 个相关结果",
            data={
                "query": request.query,
                "knowledge_base_name": request.knowledge_base_name,
                "search_type": request.search_type,
                "results": results_data,
                "total_results": len(results)
            }
        )

    except Exception as e:
        logger.error(f"查询知识库失败: {str(e)}", exc_info=True)
        return KnowledgeBaseResponse(
            success=False,
            message=f"查询失败: {str(e)}",
            data={
                "query": request.query,
                "knowledge_base_name": request.knowledge_base_name,
                "search_type": request.search_type
            }
        )

# ======================
# 对话Agent API路由
# ======================
# 3. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add]
    tool_call_count: int  # 移除累加操作，改为普通int，每次新消息重置为0
    knowledge_base_name: str  # 新增字段，用于存储当前会话使用的知识库名称
    user_document_tools: List[str]  # 新增字段，用于存储当前会话可用的用户文档工具名称
    web_search_enabled: bool  # 新增：记录web搜索是否启用

# 4. 修改模型调用节点
async def call_model(state: AgentState):
    """模型自主决策是否需要调用工具，包含参数验证和状态更新"""
    messages = state["messages"]
    knowledge_base_name = state.get("knowledge_base_name", "nsrl_tech_docs")
    
    # =============== 新增：详细日志记录 ===============
    chat_logger.info(f"🧠 模型开始思考 - 消息数量: {len(messages)}")
    if messages and isinstance(messages[-1], HumanMessage):
        chat_logger.info(f"💭 用户问题: {messages[-1].content}")
    # =============== 新增结束 ===============
    
    # 显存优化：限制会话长度，防止显存累积
    if len(messages) > 25:  # 增加限制到25条消息
        # 保留最新的20条消息和系统提示，确保不丢失重要上下文
        messages = messages[-20:]
        logger.info("⚠️ 会话历史过长，已截断以节省显存")
        chat_logger.info(f"⚠️ 会话历史过长，已截断至 {len(messages)} 条消息")
        chat_logger.info(f"📝 保留的消息类型: {[type(msg).__name__ for msg in messages]}")
        chat_logger.info(f"📝 保留的消息内容预览:")
        for i, msg in enumerate(messages[-5:], 1):  # 显示最后5条消息的预览
            if hasattr(msg, 'content') and msg.content:
                preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                chat_logger.info(f"   📝 消息 {i}: {type(msg).__name__} - {preview}")
    
    # 动态获取当前知识库的RAG工具
    rag_tool = get_rag_tool(knowledge_base_name)
    available_tools = [rag_tool]
    
    # 联网搜索功能已禁用
    chat_logger.info(f"⚠️ 联网搜索功能已禁用")
    
    # 添加用户文档工具（如果有）
    user_document_tools_list = state.get("user_document_tools", [])
    chat_logger.info(f"🔍 用户文档工具列表: {user_document_tools_list}")
    
    for tool_name in user_document_tools_list:
        tool_info = get_user_document_tool(tool_name)
        chat_logger.info(f"🔍 获取工具 {tool_name}: {tool_info}")
        if tool_info and "tool" in tool_info:
            available_tools.append(tool_info["tool"])
            chat_logger.info(f"✅ 成功添加工具: {tool_info['tool'].name if hasattr(tool_info['tool'], 'name') else '未知名称'}")
        else:
            chat_logger.warning(f"⚠️ 工具 {tool_name} 获取失败或格式不正确")
    
    chat_logger.info(f"🔧 最终可用工具: {[tool.name if hasattr(tool, 'name') else str(tool) for tool in available_tools]}")
    
    # =============== 显存优化：获取DeepSeek API模型 ===============
    logger.info("准备使用DeepSeek API模型处理对话...")
    chat_logger.info(f"🤖 获取DeepSeek API模型...")
    await gpu_resource_manager.acquire("ollama")
    try:
        # 每次调用都添加系统提示，确保工具描述完整
        chat_logger.info(f"🔄 构建系统提示和工具描述")
        # 构建工具列表描述
        tools_description = f"""1. rag_knowledge_search: 查询NSRL综合知识库（包含技术、管理、财务等全方位内容）
        - 必须参数: query (string)
        - 当前知识库: {knowledge_base_name}
        - 调用示例: {{"name": "rag_knowledge_search", "arguments": {{"query": "实验费用标准"}}}}
        - 知识库内容涵盖：
          * 实验线站技术参数和使用指南
          * NSRL管理规定和制度文件
          * 财务政策和收费标准
          * 安全防护和操作规范
          * 设备维护和故障处理
          * 用户服务和申请流程
          * 技术培训和操作手册
        - 特别说明: 知识库包含两种类型的内容：
          * QA对知识库: 以"问题：...答案：..."格式返回完整问答对，问题权重更高
          * 文档片段: 返回相关文档内容片段
        - 当检索到QA对时，系统会优先返回问题匹配度高的结果，并标记为"QA对知识库"
        - 注意: 系统仅使用本地知识库，不提供联网搜索功能"""
        # 添加用户文档工具描述
        if user_document_tools_list:
            tools_description += "\n用户上传的文档搜索工具:"
            for tool_name in user_document_tools_list:
                tool_info = get_user_document_tool(tool_name)
                if tool_info:
                    tools_description += f"\n{tool_info['tool'].name}: {tool_info['tool'].description}"
                    tools_description += "\n   - 必须参数: query (string)"
                    chat_logger.info(f"📋 添加用户文档工具描述: {tool_info['tool'].name} - {tool_info['tool'].description}")
        
        # 添加QA对说明
        tools_description += f"""
        3. 知识库内容说明:
        - 知识库 '{knowledge_base_name}' 包含PDF文档和问答对
        - PDF文档: 按原始文件名.md存储
        - 问答对: 按用户指定的文档名.md存储（如test.md）
        - 所有内容都可通过rag_knowledge_search统一搜索"""
        
        # 构建系统提示
        system_prompt = f"""你是NSRL（国家同步辐射实验室）综合智能助手，专门回答关于NSRL的各种问题，包括但不限于：

## 主要服务领域
1. **实验线站推荐**：根据用户实验需求推荐合适的同步辐射实验线站
2. **技术咨询**：回答同步辐射技术、实验方法、设备使用等技术问题
3. **管理规定**：解答NSRL的管理制度、使用规范、安全规定等
4. **财务政策**：回答关于实验费用、收费标准、财务流程等问题
5. **申请流程**：指导用户如何申请实验时间、提交提案等
6. **安全防护**：提供辐射安全、实验安全等相关指导
7. **设备维护**：回答设备状态、维护计划、故障处理等问题
8. **用户服务**：解答用户服务、技术支持、培训等相关问题

## 线站技术分类总表
以下是可用的实验线站及其技术参数：

| 实验技术类别 | 线站名称 | 能量范围 | 可用于的学科 | 能量分辨率 |
| :--- | :--- | :--- | :--- | :--- |
| **光电离质谱技术** | 质谱分析线站 (SVUV-PIMS) | 5~24.5 eV | 有机合成、石油化工、环境监测、生物化学、生物技术、临床分析、新陈代谢 | 575 @ 16 eV |
| ^^ | 燃烧光束线站 | 5~24.5 eV | 燃烧反应动力学、能源动力系统、污染物控制、生物燃料评估 | 3900 @ 7.9 eV; 4200 @ 14.6 eV |
| **红外光谱技术** | 红外谱学和显微成像光束线站 | 20-8000 cm⁻¹ (光谱)<br>700-8000 cm⁻¹ (显微) | 凝聚态物理、化学反应、材料科学、高分子材料、生命科学、医学、地学、环境、古生物学、人文考古 | 0.2 cm⁻¹ |
| **软X射线散射技术** | 共振软X射线散射线站 (RSoXS) | 220-700 eV | 有机光电、有机热电、离子交换膜等软物质材料 | 1619 @ 244.4 eV |
| **软X射线成像技术** | 软X射线成像线站 | 260-800 eV | 生命科学（细胞成像）、材料科学、能源、催化 | 500 @ 520 eV |
| **软X射线吸收光谱技术** | 软X射线磁圆二色光束线站 | 50-1000 eV | 材料科学、物理学、磁性材料研究 | 2000 @ 244 eV |
| **光电子能谱技术** | BL10B光束线 (软X射线谱学) | 100-1000 eV | 材料科学、化学、表面科学 | E/ΔE > 1000 |
| ^^ | 角分辨光电子能谱线站 (ARPES) | 7-40 eV | 凝聚态物理（高温超导、拓扑绝缘体、石墨烯） | 10000 @ 14.6 eV |
| **表面科学与催化技术** | 催化与表面科学光束线 (BL11U) | 20-600 eV | 催化科学、表面科学、半导体材料、纳米材料 | 15000 @ 29 eV |
| **光谱计量技术** | 光谱辐射标准和计量光束线 (计量线) | 1.2-200 nm<br>(约 6.2 - 1033 eV) | 光学计量、探测器与光学元件性能测试 | < 1/1000 (Δλ/λ) |
| **原子分子光谱技术** | 原子分子物理光束线站 | 7-124 eV | 原子分子物理、团簇科学、大气气溶胶科学、化学动力学 | 3000 @ 15 eV |

可用工具：
{tools_description}

工作流程：
1. 对于所有NSRL相关问题，必须首先使用 rag_knowledge_search 搜索知识库
2. 检查返回结果的最高相似度：
   * 如果最高相似度 ≥ 0.3，结果相关，基于此生成回答
   * 如果最高相似度 < 0.3，结果不相关，明确告知用户知识库中没有相关信息
3. 严格基于知识库内容回答，不得编造或推测信息

重要指导原则:
1. 回答必须严格基于知识库内容，不得编造信息
2. 如果知识库中没有相关信息，明确告知用户"知识库中没有找到相关信息"
3. 提供信息时要注明来源（来自知识库）
4. 对于技术建议，必须基于知识库中的权威资料
5. 如果知识库搜索结果相似度 < 0.3，不得基于低相似度结果生成回答
6. 回答要专业、准确、严谨
7. 如果已经调用工具超过3次仍未找到相关信息，明确告知用户知识库中无相关信息
工具调用格式要求:
- 仅使用指定的工具名称
- 仅传递工具定义中要求的参数
- 绝对不要添加额外参数（如"using"、"reason"等）
- 严格按照JSON格式输出工具调用
- 例如: {{"name": "rag_knowledge_search", "arguments": {{"query": "你的查询"}}}}
- 重要: 不要在工具调用中包含任何额外文本、解释或<think>标签
- 工具调用必须是纯JSON格式，不能有其他内容
- 错误示例: {{"name": "rag_knowledge_search", "arguments": {{"query": "...", "using": "..."}}}}"""
            
        messages = [SystemMessage(content=system_prompt)] + messages
        
        # 检查工具调用次数 - 如果超过限制，强制模型提供答案
        # 注意：tool_call_count 会在每次新消息时重置为0，所以这里的检查是针对单次对话的工具调用次数
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
                if highest_similarity < 0.3:
                    logger.warning(f"检测到低相关性结果 (相似度: {highest_similarity:.4f})，强制使用网络搜索")
                    messages.append(SystemMessage(
                        content="⚠️ 重要提示：本地知识库搜索结果相关性较低（最高相似度: {:.4f}）。"
                                "请使用web_search_tool获取最新互联网信息。".format(highest_similarity)
                    ))
        
        # 始终绑定所有可用工具
        model = gpu_resource_manager.get_ollama_model()
        chat_logger.info(f"🤖 获取到模型类型: {type(model).__name__}")
        model_with_tools = model.bind_tools(available_tools)
        
        chat_logger.info(f"🤖 开始调用DeepSeek API模型...")
        # 调用模型
        response = await model_with_tools.ainvoke(messages)
        chat_logger.info(f"✅ 模型响应完成")
        
        # 记录模型响应内容
        if hasattr(response, "content") and response.content:
            # 记录完整的模型回答内容
            chat_logger.info(f"💬 模型回答完整内容:")
            chat_logger.info(f"📝 {response.content}")
            # 同时记录长度信息
            chat_logger.info(f"📊 回答长度: {len(response.content)} 字符")
        
        # 记录工具调用
        if hasattr(response, "tool_calls") and response.tool_calls:
            chat_logger.info(f"🔧 模型决定调用工具: {len(response.tool_calls)} 个")
            for i, tool_call in enumerate(response.tool_calls):
                chat_logger.info(f"  🔧 工具 {i+1}: {tool_call['name']} - 参数: {tool_call['args']}")
        else:
            # 尝试从文本内容中解析工具调用
            import re
            import json
            content = response.content if hasattr(response, "content") else ""
            
            # 匹配工具调用格式：{"name": "tool_name", "arguments": {...}}
            tool_call_pattern = r'\{[^}]*"name"\s*:\s*"([^"]+)"[^}]*"arguments"\s*:\s*(\{[^}]*\})[^}]*\}'
            tool_calls = []
            
            for match in re.finditer(tool_call_pattern, content):
                tool_name = match.group(1)
                try:
                    tool_args = json.loads(match.group(2))
                    tool_calls.append({
                        "name": tool_name,
                        "args": tool_args,
                        "id": f"call_{len(tool_calls)}"
                    })
                    chat_logger.info(f"🔍 从文本解析到工具调用: {tool_name} - 参数: {tool_args}")
                except json.JSONDecodeError:
                    chat_logger.warning(f"⚠️ 工具调用参数JSON解析失败: {match.group(2)}")
            
            if tool_calls:
                # 将解析的工具调用添加到response对象
                response.tool_calls = tool_calls
                chat_logger.info(f"🔧 从文本解析到 {len(tool_calls)} 个工具调用")
            else:
                chat_logger.info(f"💬 模型直接回答，无工具调用")
        
        # 关键修复：验证并清理工具调用参数
        if hasattr(response, "tool_calls") and response.tool_calls:
            chat_logger.info(f"🧹 开始清理工具调用参数...")
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
                        chat_logger.warning(f"⚠️ rag_knowledge_search缺少query参数，使用备用参数: {first_arg}")
                elif tool_call["name"] == "web_search_tool":
                    # 仅保留query参数
                    if "query" in tool_call["args"]:
                        valid_args["query"] = tool_call["args"]["query"]
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
            # 替换原始的tool_calls
            response.tool_calls = cleaned_tool_calls
            logger.info(f"已清理工具调用参数，移除无效参数")
            chat_logger.info(f"✅ 工具调用参数清理完成")
        
        # 计算工具调用增量
        tool_call_increment = 1 if (hasattr(response, "tool_calls") and response.tool_calls) else 0
        
        # =============== 新增：详细日志记录 ===============
        chat_logger.info(f"📊 工具调用统计 - 本次增量: {tool_call_increment}")
        chat_logger.info(f"🎯 模型思考完成，准备返回结果")
        # =============== 新增结束 ===============
        
        # 确保web_search_enabled状态正确传递
        web_search_enabled = state.get("web_search_enabled", True)
        chat_logger.info(f"📤 返回状态 - Web搜索状态: {'启用' if web_search_enabled else '禁用'}")
        
        return {
            "messages": [response],
            "tool_call_count": tool_call_increment,  # 这里会累加，但每次新消息会重置
            "knowledge_base_name": knowledge_base_name,  # 确保传递知识库名称
            "user_document_tools": user_document_tools_list,  # 确保传递用户文档工具列表
            "web_search_enabled": web_search_enabled  # 确保状态正确传递
        }
    finally:
        await gpu_resource_manager.release()
        logger.info("DeepSeek API模型处理完成，资源已释放")
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
    # 注意：tool_call_count 会在每次新消息时重置为0，所以这里的检查是针对单次对话的工具调用次数
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
    knowledge_base_name = state.get("knowledge_base_name", "nsrl_tech_docs")
    
    # =============== 新增：详细日志记录 ===============
    chat_logger.info(f"🔧 开始执行工具节点 - 知识库: {knowledge_base_name}")
    chat_logger.info(f"📝 需要执行的工具调用: {len(last_message.tool_calls)} 个")
    for i, tool_call in enumerate(last_message.tool_calls):
        chat_logger.info(f"  🔧 工具 {i+1}: {tool_call['name']} - 参数: {tool_call['args']}")
    # =============== 新增结束 ===============
    
    # 动态获取当前知识库的RAG工具
    rag_tool = get_rag_tool(knowledge_base_name)
    # 创建工具映射
    tools = {
        "rag_knowledge_search": rag_tool
    }
    # 联网搜索功能已禁用
    chat_logger.info(f"⚠️ 工具节点 - 联网搜索功能已禁用")
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
                chat_logger.info(f"🔧 调用工具 {tool_name} 参数: {tool_call['args']}")
                response = tool.invoke(tool_call["args"])
                
                # 记录工具返回结果
                response_str = str(response)
                chat_logger.info(f"✅ 工具 {tool_name} 执行成功: {len(response_str)}字符")
                
                # =============== 新增：记录工具返回的完整内容 ===============
                chat_logger.info(f"📤 工具 {tool_name} 返回内容:")
                chat_logger.info(f"📝 {response_str}")
                chat_logger.info(f"📊 工具返回内容长度: {len(response_str)} 字符")
                # =============== 新增结束 ===============
                
                # 如果是RAG工具，记录相似度信息
                if tool_name == "rag_knowledge_search" and "相似度:" in response_str:
                    similarities = re.findall(r"相似度: ([\d.]+)", response_str)
                    if similarities:
                        max_sim = max(float(s) for s in similarities)
                        chat_logger.info(f"🎯 RAG工具最高相似度: {max_sim:.4f}")
                
                # 在工具结果中包含用户的原始问题
                user_question = ""
                for msg in reversed(state["messages"]):
                    if isinstance(msg, HumanMessage):
                        user_question = msg.content
                        break
                
                # 构建包含用户问题的工具响应
                enhanced_content = f"用户问题: {user_question}\n\n工具结果:\n{str(response)}"
                
                # =============== 新增：详细记录工具返回结果 ===============
                chat_logger.info(f"🔧 工具 {tool_name} 执行成功")
                chat_logger.info(f"📤 工具返回内容长度: {len(str(response))} 字符")
                chat_logger.info(f"📝 工具返回内容预览: {str(response)[:200]}...")
                chat_logger.info(f"🎯 工具返回完整内容:")
                chat_logger.info(f"{str(response)}")
                chat_logger.info(f"🔧 工具 {tool_name} 返回记录完成")
                # =============== 新增结束 ===============
                
                outputs.append(
                    ToolMessage(
                        content=enhanced_content,
                        name=tool_name,
                        tool_call_id=tool_call["id"]
                    )
                )
            except Exception as e:
                chat_logger.error(f"❌ 工具 {tool_name} 执行失败: {str(e)}")
                # =============== 新增：记录工具执行失败的详细信息 ===============
                chat_logger.error(f"📤 工具 {tool_name} 执行失败详情:")
                chat_logger.error(f"📝 错误信息: {str(e)}")
                chat_logger.error(f"🔧 工具参数: {tool_call['args']}")
                chat_logger.error(f"📊 错误类型: {type(e).__name__}")
                # =============== 新增结束 ===============
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
            # =============== 新增：记录工具不存在的详细信息 ===============
            chat_logger.error(f"📤 工具 {tool_name} 不存在详情:")
            chat_logger.error(f"🔧 请求的工具名称: {tool_name}")
            chat_logger.error(f"📋 可用工具列表: {list(tools.keys())}")
            chat_logger.error(f"📊 可用工具数量: {len(tools.keys())}")
            # =============== 新增结束 ===============
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
    conversation_id: Optional[str] = None  # 新增：对话ID，用于跨设备同步
    stream: Optional[bool] = False  # 是否启用流式响应
    knowledge_base_name: Optional[str] = "nsrl_tech_docs"  # 新增参数，默认为"nsrl_tech_docs"
    url: Optional[str] = None
    enable_web_search: Optional[bool] = True
    message_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    conversation_id: Optional[str] = None  # 新增：对话ID
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
async def chat_endpoint(request: ChatRequest, http_request: Request = None):
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
    # 获取当前用户信息
    current_user = get_current_user_from_token(http_request) if http_request else None
    user_id = current_user.id if current_user else "anonymous"
    
    # 使用用户ID + 会话名称生成session_id，确保跨设备一致性
    if request.session_id:
        session_id = f"user_{user_id}_{request.session_id}"
    else:
        # 默认会话使用用户ID
        session_id = f"user_{user_id}_default"
    
    # 处理对话ID
    conversation_id = request.conversation_id
    if not conversation_id:
        # 如果没有提供对话ID，生成一个新的
        conversation_id = f"conv_{user_id}_{int(time.time())}"
    
    config = {"configurable": {"thread_id": session_id}}
    
    # =============== 新增：详细日志记录 ===============
    chat_logger.info(f"🚀 开始处理聊天请求 - Session: {session_id}")
    chat_logger.info(f"📝 用户输入: {request.message}")
    chat_logger.info(f"🔧 知识库: {request.knowledge_base_name}")
    chat_logger.info(f"🌐 Web搜索: {'启用' if request.enable_web_search else '禁用'}")
    if request.url:
        chat_logger.info(f"📄 文档URL: {request.url}")
    # =============== 新增结束 ===============
    
    try:
        # 获取当前会话状态
        state = await graph.aget_state(config)
        chat_logger.info(f"🔍 调试 - 获取到的状态: {state}")
        chat_logger.info(f"🔍 调试 - session_id: {session_id}")
        chat_logger.info(f"🔍 调试 - config: {config}")
        if state:
            chat_logger.info(f"🔍 调试 - state.values类型: {type(state.values)}")
            chat_logger.info(f"🔍 调试 - state.values内容: {state.values}")
            chat_logger.info(f"🔍 调试 - 是否有messages: {'messages' in state.values}")
            if 'messages' in state.values:
                chat_logger.info(f"🔍 调试 - messages数量: {len(state.values['messages'])}")
                for i, msg in enumerate(state.values['messages']):
                    chat_logger.info(f"🔍 调试 - 消息{i}: {type(msg).__name__} - {msg.content[:50]}...")
        else:
            chat_logger.info(f"🔍 调试 - 没有找到历史状态，这是新会话")
        
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
                    tool_name = await register_user_document_tool(
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
            chat_logger.info(f"🆕 新会话 - 请求中的enable_web_search: {request.enable_web_search}")
            initial_state = {
                "messages": [HumanMessage(content=request.message)],
                "knowledge_base_name": request.knowledge_base_name,
                "tool_call_count": 0,
                "user_document_tools": user_document_tools_list,
                "web_search_enabled": request.enable_web_search  # 保存web搜索开关状态
            }
            chat_logger.info(f"🆕 新会话 - 初始状态中的web_search_enabled: {initial_state['web_search_enabled']}")
        else:
            # 续会话 - 复制现有状态并添加新消息
            chat_logger.info(f"🔄 继续现有会话，历史消息数: {len(state.values.get('messages', []))}")
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
                    tool_name = await register_user_document_tool(
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
            chat_logger.info(f"🔍 调试 - 历史状态中的web_search_enabled: {state.values.get('web_search_enabled', '未设置')}")
            chat_logger.info(f"🔍 调试 - 请求中的enable_web_search: {request.enable_web_search}")
            
            # 临时修复：优先使用请求参数，确保web搜索状态正确传递
            if request.enable_web_search is not None:
                web_search_enabled = request.enable_web_search
                chat_logger.info(f"🔧 临时修复 - 使用请求参数: {web_search_enabled}")
            else:
                web_search_enabled = state.values.get("web_search_enabled", True)
                chat_logger.info(f"🔧 临时修复 - 使用历史状态: {web_search_enabled}")
            
            chat_logger.info(f"🔄 续会话 - Web搜索状态: {'启用' if web_search_enabled else '禁用'}")
            initial_state = {
                "messages": state.values["messages"] + [HumanMessage(content=request.message)],
                "knowledge_base_name": knowledge_base_name,
                "tool_call_count": 0,  # 每次新消息都重置工具调用次数
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
        
        # =============== 新增：记录正常输出的详细信息 ===============
        chat_logger.info(f"🚀 ====== 正常输出API返回内容 ======")
        chat_logger.info(f"🆔 会话ID: {session_id}")
        chat_logger.info(f"📤 返回的最终回答:")
        chat_logger.info(f"   📄 回答长度: {len(last_msg.content)} 字符")
        chat_logger.info(f"   📝 完整回答内容: {last_msg.content}")
        chat_logger.info(f"📊 返回的对话历史:")
        chat_logger.info(f"   📝 历史记录数量: {len(history)}")
        for i, hist in enumerate(history):
            # 记录完整的对话历史内容
            chat_logger.info(f"     📤 历史 {i+1} [{hist['role']}]:")
            chat_logger.info(f"        📝 完整内容: {hist['content']}")
            chat_logger.info(f"        📊 内容长度: {len(hist['content'])} 字符")
            # 同时保留预览信息
            chat_logger.info(f"        📄 内容预览: {hist['content'][:100]}{'...' if len(hist['content']) > 100 else ''}")
        chat_logger.info(f"🔧 返回的工具调用信息:")
        if tool_calls:
            for i, tool_call in enumerate(tool_calls):
                chat_logger.info(f"   🔧 工具 {i+1}: {tool_call['name']} - 参数: {tool_calls[i]['args']}")
        else:
            chat_logger.info(f"   ℹ️ 无工具调用")
        chat_logger.info(f"🚀 ====== 正常输出API返回内容结束 ======")
        # =============== 新增结束 ===============
        
        chat_logger.info(f"📤 返回最终回答，长度: {len(last_msg.content)}")
        chat_logger.info(f"🎯 对话完成 - Session: {session_id}")
        
        # 保存聊天日志到文件
        try:
            chat_log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "session_id": session_id,
                "user_message": request.message,
                "assistant_response": last_msg.content,
                "knowledge_base": request.knowledge_base_name,
                "web_search_enabled": request.enable_web_search,
                "tool_calls": tool_calls if tool_calls else [],
                "conversation_length": len(history)
            }
            
            # 保存到聊天日志文件
            chat_log_file = os.path.join('logs', f"chat_logs_{datetime.datetime.now().strftime('%Y%m%d')}.log")
            with open(chat_log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{chat_log_entry['timestamp']}] Session: {chat_log_entry['session_id']}\n")
                f.write(f"User: {chat_log_entry['user_message']}\n")
                f.write(f"Assistant: {chat_log_entry['assistant_response']}\n")
                f.write(f"Knowledge Base: {chat_log_entry['knowledge_base']}\n")
                f.write(f"Web Search: {'Enabled' if chat_log_entry['web_search_enabled'] else 'Disabled'}\n")
                f.write(f"Tool Calls: {len(chat_log_entry['tool_calls'])}\n")
                f.write(f"Conversation Length: {chat_log_entry['conversation_length']}\n")
                f.write("-" * 80 + "\n\n")
            
            chat_logger.info(f"💾 聊天日志已保存到: {chat_log_file}")
        except Exception as e:
            chat_logger.error(f"❌ 保存聊天日志失败: {str(e)}")
        
        return ChatResponse(
            response=last_msg.content,
            session_id=session_id,
            conversation_id=conversation_id,
            conversation_history=history,
            tool_calls=tool_calls if tool_calls else None
        )
    except HTTPException:
        raise
    except Exception as e:
        chat_logger.error(f"❌ 处理请求失败: {str(e)}")
        logger.error(f"❌ 处理请求失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"请求处理失败: {str(e)}"
        )

# 9. 流式响应API（可选）
@agent_router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, http_request: Request = None):
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
    # 获取当前用户信息
    current_user = get_current_user_from_token(http_request) if http_request else None
    user_id = current_user.id if current_user else "anonymous"
    
    # 使用用户ID + 会话名称生成session_id，确保跨设备一致性
    if request.session_id:
        session_id = f"user_{user_id}_{request.session_id}"
    else:
        # 默认会话使用用户ID
        session_id = f"user_{user_id}_default"
    
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
                        tool_name = await register_user_document_tool(
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
                
                # 应用临时修复：优先使用请求参数，确保web搜索状态正确传递
                chat_logger.info(f"🔍 流式API调试 - 历史状态中的web_search_enabled: {state.values.get('web_search_enabled', '未设置')}")
                chat_logger.info(f"🔍 流式API调试 - 请求中的enable_web_search: {request.enable_web_search}")
                
                if request.enable_web_search is not None:
                    web_search_enabled = request.enable_web_search
                    chat_logger.info(f"🔧 流式API临时修复 - 使用请求参数: {web_search_enabled}")
                else:
                    web_search_enabled = state.values.get("web_search_enabled", True)
                    chat_logger.info(f"🔧 流式API临时修复 - 使用历史状态: {web_search_enabled}")
                
                chat_logger.info(f"🔄 流式API续会话 - Web搜索状态: {'启用' if web_search_enabled else '禁用'}")
                if request.url and not any(
                        tool_name.startswith("search_" + session_id) for tool_name in user_document_tools_list):
                    try:
                        logger.info(f"处理用户上传的文档URL: {request.url}")
                        document_id = f"{session_id}_{uuid.uuid4().hex[:8]}"
                        tool_name = await register_user_document_tool(
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
                    "tool_call_count": 0,  # 每次新消息都重置工具调用次数
                    "user_document_tools": user_document_tools_list,
                    "web_search_enabled": web_search_enabled
                }
            
            # 优化分块大小和流式处理逻辑
            CHUNK_SIZE = 50  # 增大分块大小，减少过度分割
            full_text = ""
            last_sent_length = 0  # 记录已发送的文本长度
            
            logger.info(f"🚀 开始流式响应，分块大小: {CHUNK_SIZE}")
            
            # =============== 新增：记录流式输出开始信息 ===============
            chat_logger.info(f"🌊 ====== 流式输出API开始 ======")
            chat_logger.info(f"🆔 会话ID: {session_id}")
            chat_logger.info(f"🆔 消息ID: {message_id}")
            chat_logger.info(f"📝 用户输入: {request.message}")
            chat_logger.info(f"🔧 知识库名称: {request.knowledge_base_name}")
            chat_logger.info(f"🌐 Web搜索启用: {request.enable_web_search}")
            if request.url:
                chat_logger.info(f"📄 文档URL: {request.url}")
            # =============== 新增结束 ===============
            
            # 发送开始标记
            start_data = {
                "text": "",
                "finish_reason": "start",
                "session_id": session_id,
                "message_id": message_id
            }
            yield f"data: {json.dumps(start_data)}\n"
            await asyncio.sleep(0.01)
            
            try:
                # 按照你的思路：在astream_log过程中实时检测对话结束信号
                chat_logger.info(f"🔍 开始实时检测对话流程...")
                
                # 用于跟踪对话状态
                thinking_content = ""
                tool_results = []
                final_answer = ""
                conversation_ended = False
                
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
                        
                        # =============== 方案一：修复路径匹配逻辑 ===============
                        
                        # 记录关键路径，包括工具调用相关的路径
                        if value is not None and ("final_output" in path or "call_model" in path or "tools" in path or "search_" in path):
                            chat_logger.info(f"路径: {path}")
                        
                        # 1. 处理模型调用日志（思考过程和最终回答）- 更精确的路径匹配
                        if (path.startswith("/logs/call_model/") or 
                            path.startswith("/state/messages") or
                            path.endswith("/final_output") or
                            path.endswith("/streamed_output/-") or
                            "call_model" in path) and value is not None:
                            if isinstance(value, dict) and "messages" in value:
                                for msg in value["messages"]:
                                    if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
                                        content = msg.content
                                        
                                        # 检查是否包含<think>标签
                                        if "<think>" in content:
                                            # 这是思考过程
                                            thinking_content = content
                                            chat_logger.info(f"💭 思考过程: {len(content)}字符")
                                            
                                            # 实时发送思考过程
                                            if content not in full_text:
                                                # 分块发送思考内容
                                                content_chunks = [content[i:i+CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
                                                for chunk in content_chunks:
                                                    data = {
                                                        "text": chunk,
                                                        "finish_reason": None,
                                                        "session_id": session_id,
                                                        "message_id": message_id
                                                    }
                                                    yield f"data: {json.dumps(data)}\n"
                                                    await asyncio.sleep(0.01)
                                                
                                                full_text += content
                                                last_sent_length = len(full_text)
                                                chat_logger.info(f"📤 思考过程已发送")
                                        else:
                                            # 这是最终回答（不包含<think>标签）
                                            final_answer = content
                                            chat_logger.info(f"🎯 最终回答: {len(content)}字符")
                                            
                                            # 检查是否已经发送过这个回答
                                            if content not in full_text:
                                                # 分块发送最终回答
                                                answer_chunks = [content[i:i+CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
                                                for chunk in answer_chunks:
                                                    data = {
                                                        "text": chunk,
                                                        "finish_reason": None,
                                                        "session_id": session_id,
                                                        "message_id": message_id
                                                    }
                                                    yield f"data: {json.dumps(data)}\n"
                                                    await asyncio.sleep(0.01)
                                                
                                                full_text += content
                                                last_sent_length = len(full_text)
                                                chat_logger.info(f"✅ 最终回答已发送")
                                                
                                                # 检测到最终回答后，标记对话即将结束
                                                conversation_ended = True
                                                chat_logger.info(f"🏁 对话即将结束")
                        
                        # 2. 处理工具调用结果（从tool_node的执行结果中获取）- 更精确的路径匹配
                        elif (path.startswith("/logs/tools/") or 
                              path.startswith("/logs/") or
                              "tools" in path or
                              "tool_node" in path or
                              "search_" in path or
                              path.endswith("/final_output")) and value is not None:
                            # 检查是否是工具执行结果
                            if isinstance(value, dict) and "messages" in value:
                                for msg in value["messages"]:
                                    if isinstance(msg, ToolMessage) and hasattr(msg, "content") and msg.content:
                                        tool_content = msg.content
                                        tool_name = getattr(msg, "name", "未知工具")
                                        
                                        # 清理工具调用结果内容
                                        if isinstance(tool_content, str):
                                            import re
                                            clean_content = re.sub(r'<[^>]+>', '', tool_content)
                                            clean_content = re.sub(r'https?://[^\s]+', '', clean_content)
                                            clean_content = re.sub(r'\n\s*\n', '\n', clean_content)
                                            clean_content = clean_content.strip()
                                        else:
                                            clean_content = str(tool_content)
                                        
                                        # 检查是否已经发送过这个工具结果
                                        if clean_content not in [result["content"] for result in tool_results]:
                                            tool_results.append({
                                                "name": tool_name,
                                                "content": clean_content
                                            })
                                            
                                            # 添加工具调用结果的醒目标识
                                            tool_result_content = f"\n\n{'='*50}\n🔧 工具调用结果: {tool_name}\n{'='*50}\n{clean_content}\n{'='*50}\n"
                                            
                                            # 分块发送工具调用结果
                                            tool_chunks = [tool_result_content[i:i+CHUNK_SIZE] for i in range(0, len(tool_result_content), CHUNK_SIZE)]
                                            for chunk in tool_chunks:
                                                data = {
                                                    "text": chunk,
                                                    "finish_reason": None,
                                                    "session_id": session_id,
                                                    "message_id": message_id
                                                }
                                                yield f"data: {json.dumps(data)}\n"
                                                await asyncio.sleep(0.01)
                                            
                                            full_text += clean_content
                                            last_sent_length = len(full_text)
                                            chat_logger.info(f"🔧 工具调用结果已发送: {tool_name}, 长度: {len(clean_content)}")
                                            # =============== 新增：记录流式输出中的工具返回完整内容 ===============
                                            chat_logger.info(f"📤 流式输出 - 工具 {tool_name} 返回内容:")
                                            chat_logger.info(f"📝 {clean_content}")
                                            chat_logger.info(f"📊 流式输出 - 工具返回内容长度: {len(clean_content)} 字符")
                                            # =============== 新增结束 ===============
                                        
                                        # 检查是否包含对话结束信号
                                        if "无工具调用，对话结束" in str(msg.content):
                                            conversation_ended = True
                                            chat_logger.info(f"🏁 从工具消息中检测到对话结束信号")
                                    
                                    # 检查AI消息中是否包含对话结束信号
                                    elif isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
                                        if "无工具调用，对话结束" in str(msg.content):
                                            conversation_ended = True
                                            chat_logger.info(f"🏁 从AI消息中检测到对话结束信号")
                        

                        
                        # 3. 检测对话结束信号 - 更精确的路径匹配
                        elif (path.startswith("/logs/should_continue/") or
                              path.startswith("/logs/") or
                              "should_continue" in path or
                              "END" in str(value) or
                              "end" in path.lower()) and value is not None:
                            # 检查是否包含"无工具调用，对话结束"的信号
                            if isinstance(value, str) and "无工具调用，对话结束" in value:
                                conversation_ended = True
                                chat_logger.info(f"🏁 检测到对话结束信号: {value}")
                            elif isinstance(value, dict) and value.get("messages"):
                                # 检查消息中是否包含结束信号
                                for msg in value["messages"]:
                                    if hasattr(msg, "content") and msg.content:
                                        if "无工具调用，对话结束" in str(msg.content):
                                            conversation_ended = True
                                            chat_logger.info(f"🏁 从消息中检测到对话结束信号")
                                            break
                        
                        # 4. 专门检测工具调用结果 - 更宽泛的检测
                        if value is not None:
                            # 检查是否包含ToolMessage，无论路径如何
                            if isinstance(value, dict) and "messages" in value:
                                chat_logger.info(f"🔍 检查路径 {path} 的messages字段，消息数量: {len(value['messages'])}")
                                for i, msg in enumerate(value["messages"]):
                                    chat_logger.info(f"🔍 消息 {i}: 类型={type(msg).__name__}, 内容长度={len(str(msg.content)) if hasattr(msg, 'content') else 'N/A'}")
                                    if isinstance(msg, ToolMessage) and hasattr(msg, "content") and msg.content:
                                        chat_logger.info(f"🔧 发现工具调用结果: {getattr(msg, 'name', '未知工具')}, 路径: {path}")
                                        tool_content = msg.content
                                        tool_name = getattr(msg, "name", "未知工具")
                                        
                                        # 清理工具调用结果内容
                                        if isinstance(tool_content, str):
                                            import re
                                            clean_content = re.sub(r'<[^>]+>', '', tool_content)
                                            clean_content = re.sub(r'https?://[^\s]+', '', clean_content)
                                            clean_content = re.sub(r'\n\s*\n', '\n', clean_content)
                                            clean_content = clean_content.strip()
                                        else:
                                            clean_content = str(tool_content)
                                        
                                        # 检查是否已经发送过这个工具结果
                                        if clean_content not in [result["content"] for result in tool_results]:
                                            tool_results.append({
                                                "name": tool_name,
                                                "content": clean_content
                                            })
                                            
                                            # 添加工具调用结果的醒目标识
                                            tool_result_content = f"\n\n{'='*50}\n🔧 工具调用结果: {tool_name}\n{'='*50}\n{clean_content}\n{'='*50}\n"
                                            
                                            # 分块发送工具调用结果
                                            tool_chunks = [tool_result_content[i:i+CHUNK_SIZE] for i in range(0, len(tool_result_content), CHUNK_SIZE)]
                                            for chunk in tool_chunks:
                                                data = {
                                                    "text": chunk,
                                                    "finish_reason": None,
                                                    "session_id": session_id,
                                                    "message_id": message_id
                                                }
                                                yield f"data: {json.dumps(data)}\n"
                                                await asyncio.sleep(0.01)
                                            
                                            full_text += clean_content
                                            last_sent_length = len(full_text)
                                            chat_logger.info(f"🔧 工具调用结果已发送: {tool_name}, 长度: {len(clean_content)}")
                                            # =============== 新增：记录流式输出中的工具返回完整内容（备用检测） ===============
                                            chat_logger.info(f"📤 流式输出备用检测 - 工具 {tool_name} 返回内容:")
                                            chat_logger.info(f"📝 {clean_content}")
                                            chat_logger.info(f"📊 流式输出备用检测 - 工具返回内容长度: {len(clean_content)} 字符")
                                            # =============== 新增结束 ===============
                        
                        
                        # 5. 备用检测机制 - 捕获任何包含AIMessage的路径
                        if value is not None:
                            # 检查是否包含AIMessage，作为备用检测
                            if isinstance(value, dict) and "messages" in value:
                                for msg in value["messages"]:
                                    if isinstance(msg, AIMessage) and hasattr(msg, "content") and msg.content:
                                        content = msg.content
                                        # 如果不包含<think>标签，可能是最终回答
                                        if "<think>" not in content and len(content) > 50:  # 过滤掉太短的内容
                                            chat_logger.info(f"🔄 备用检测最终回答: {len(content)}字符")
                                            
                                            # 检查是否已经发送过这个内容
                                            if content not in full_text:
                                                final_answer = content
                                                
                                                # 分块发送最终回答
                                                answer_chunks = [content[i:i+CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
                                                for chunk in answer_chunks:
                                                    data = {
                                                        "text": chunk,
                                                        "finish_reason": None,
                                                        "session_id": session_id,
                                                        "message_id": message_id
                                                    }
                                                    yield f"data: {json.dumps(data)}\n"
                                                    await asyncio.sleep(0.01)
                                                
                                                full_text += content
                                                last_sent_length = len(full_text)
                                                conversation_ended = True
                                                chat_logger.info(f"✅ 备用机制已发送")
                        
                        # =============== 基于你的思路结束 ===============
                
                # 基于你的思路：检测到对话结束信号后，记录状态但不发送重复的结束信号
                if conversation_ended:
                    chat_logger.info(f"🏁 检测到对话结束，总发送长度: {len(full_text)}")
                else:
                    chat_logger.warning(f"⚠️ 未检测到对话结束信号，使用默认结束")
                
                # 记录流式输出完成信息
                chat_logger.info(f"🌊 ====== 流式输出API完成 ======")
                chat_logger.info(f"📊 流式输出统计:")
                chat_logger.info(f"   📝 总内容长度: {len(full_text)} 字符")
                chat_logger.info(f"   📤 已发送长度: {last_sent_length} 字符")
                chat_logger.info(f"   📋 分块大小: {CHUNK_SIZE}")
                chat_logger.info(f"   🏁 结束状态: {'正常结束' if conversation_ended else '默认结束'}")
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
                    logger.debug(f"📤 发送最终剩余内容，长度: {len(remaining_text)}")
                    
                    # =============== 新增：记录最终剩余内容发送 ===============
                    chat_logger.info(f"📤 流式最终剩余内容发送: 长度={len(remaining_text)}, 内容={remaining_text}")
                    # =============== 新增结束 ===============
            
            # 统一发送结束标记（避免重复）
            end_data = {
                "text": "",
                "finish_reason": "stop",
                "session_id": session_id,
                "message_id": message_id
            }
            yield f"data: {json.dumps(end_data)}\n"
            await asyncio.sleep(0.01)
            
            # =============== 新增：记录流式输出结束信息 ===============
            chat_logger.info(f"🌊 ====== 流式输出API结束 ======")
            chat_logger.info(f"🆔 会话ID: {session_id}")
            chat_logger.info(f"🆔 消息ID: {message_id}")
            chat_logger.info(f"📊 流式输出统计:")
            chat_logger.info(f"   📝 总内容长度: {len(full_text)} 字符")
            chat_logger.info(f"   📤 已发送长度: {last_sent_length} 字符")
            chat_logger.info(f"   📋 分块大小: {CHUNK_SIZE}")
            chat_logger.info(f"🌊 ====== 流式输出API结束 ======")
            
            # 保存流式聊天日志到文件
            try:
                chat_log_entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "session_id": session_id,
                    "message_id": message_id,
                    "user_message": request.message,
                    "assistant_response": full_text,
                    "knowledge_base": request.knowledge_base_name,
                    "web_search_enabled": request.enable_web_search,
                    "response_type": "streaming",
                    "total_length": len(full_text),
                    "sent_length": last_sent_length
                }
                
                # 保存到聊天日志文件
                chat_log_file = os.path.join('logs', f"chat_logs_{datetime.datetime.now().strftime('%Y%m%d')}.log")
                with open(chat_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"[{chat_log_entry['timestamp']}] Session: {chat_log_entry['session_id']} (Streaming)\n")
                    f.write(f"Message ID: {chat_log_entry['message_id']}\n")
                    f.write(f"User: {chat_log_entry['user_message']}\n")
                    f.write(f"Assistant: {chat_log_entry['assistant_response']}\n")
                    f.write(f"Knowledge Base: {chat_log_entry['knowledge_base']}\n")
                    f.write(f"Web Search: {'Enabled' if chat_log_entry['web_search_enabled'] else 'Disabled'}\n")
                    f.write(f"Response Type: {chat_log_entry['response_type']}\n")
                    f.write(f"Total Length: {chat_log_entry['total_length']}\n")
                    f.write(f"Sent Length: {chat_log_entry['sent_length']}\n")
                    f.write("-" * 80 + "\n\n")
                
                chat_logger.info(f"💾 流式聊天日志已保存到: {chat_log_file}")
            except Exception as e:
                chat_logger.error(f"❌ 保存流式聊天日志失败: {str(e)}")
            # =============== 新增结束 ===============
            
            # 最终显存清理
            import gc
            gc.collect()
            logger.info(f"🧹 流式响应完成，执行最终显存清理。总内容长度: {len(full_text)}, 已发送长度: {last_sent_length}")
            
            # =============== 最小化修复总结 ===============
            # 1. ✅ 解决了重复发送结束信号的问题（核心问题）
            # 2. 🔄 保持了原有的复杂检测逻辑不变
            # 3. 🔄 保持了所有备用检测机制
            # 4. 🔄 保持了原有的路径匹配逻辑
            # 5. ✅ 确保只发送一次结束信号
            # 
            # 说明：这是最小化修复方案，只解决重复发送问题，
            # 不改变原有的检测逻辑，降低修改风险
            # =============== 最小化修复总结结束 ===============
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
            
            # =============== 新增：记录流式输出错误信息 ===============
            chat_logger.error(f"❌ 流式输出API错误:")
            chat_logger.error(f"   🆔 会话ID: {session_id}")
            chat_logger.error(f"   🆔 消息ID: {message_id}")
            chat_logger.error(f"   ❌ 错误信息: {str(e)}")
            chat_logger.error(f"   📚 错误类型: {type(e).__name__}")
            # =============== 新增结束 ===============
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"  # 确保这是正确的媒体类型
    )

# 10. 会话管理API
@agent_router.get("/sessions")
async def get_user_sessions(request: Request):
    """获取当前用户的所有会话列表"""
    try:
        # 获取当前用户信息
        current_user = get_current_user_from_token(request)
        if not current_user:
            raise HTTPException(status_code=401, detail="未授权访问")
        
        user_id = current_user.id
        user_sessions = []
        
        # 从专门的对话历史数据库获取会话数据
        sessions = []
        try:
            # 从专门的对话历史数据库获取会话列表
            async with chat_history_pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT conversation_id, title, updated_at, message_count
                        FROM conversations 
                        WHERE user_id = %s 
                        ORDER BY updated_at DESC
                    """, (user_id,))
                    
                    rows = await cur.fetchall()
                    for row in rows:
                        conversation_id, title, updated_at, message_count = row
                        sessions.append({
                            "session_id": conversation_id,
                            "session_name": title or "新对话",
                            "last_updated": str(updated_at),
                            "message_count": message_count or 0
                        })
            
            logger.info(f"从专用数据库获取用户对话: {user_id}, 找到 {len(sessions)} 个对话")
            
        except Exception as e:
            logger.error(f"从专用数据库获取会话失败: {str(e)}")
        
        return {
            "user_id": user_id,
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"❌ 获取用户会话列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")

@agent_router.get("/conversations/{user_id}")
async def get_user_conversations(user_id: str, request: Request):
    """获取用户的所有对话列表"""
    try:
        # 验证用户权限
        current_user = get_current_user_from_token(request)
        if not current_user or str(current_user.id) != str(user_id):
            raise HTTPException(status_code=403, detail="无权限访问该用户的对话")
        
        # 从checkpointer获取用户的对话列表
        global checkpointer
        conversations = []
        
        if checkpointer:
            try:
                # 这里应该查询数据库中所有属于该用户的对话
                # 由于checkpointer的限制，我们暂时使用已知的会话ID模式
                # 在实际应用中，应该有一个专门的conversations表来存储对话元数据
                
                # 检查默认会话
                default_session_id = f"user_{user_id}_default"
                config = {"configurable": {"thread_id": default_session_id}}
                state = await checkpointer.aget_state(config)
                
                if state and state.values and "messages" in state.values:
                    messages = state.values["messages"]
                    if messages and len(messages) > 0:
                        # 获取第一条用户消息作为标题
                        first_user_msg = None
                        for msg in messages:
                            if hasattr(msg, 'type') and msg.type == 'human':
                                first_user_msg = msg.content
                                break
                        
                        title = first_user_msg[:30] + "..." if first_user_msg and len(first_user_msg) > 30 else (first_user_msg or "新对话")
                        
                        conversations.append({
                            "conversation_id": default_session_id,
                            "title": title,
                            "created_at": state.config.get("configurable", {}).get("checkpoint_id", ""),
                            "message_count": len(messages)
                        })
            except Exception as e:
                logger.error(f"从checkpointer获取对话失败: {str(e)}")
        
        logger.info(f"从数据库获取用户对话: {user_id}, 找到 {len(conversations)} 个对话")
        
        return {
            "status": "success",
            "user_id": user_id,
            "conversations": conversations
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户对话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取用户对话失败: {str(e)}")

@agent_router.get("/conversations/{user_id}/{conversation_id}")
async def get_conversation_history(user_id: str, conversation_id: str, request: Request):
    """获取特定对话的历史记录"""
    try:
        # 验证用户权限
        current_user = get_current_user_from_token(request)
        if not current_user or str(current_user.id) != str(user_id):
            raise HTTPException(status_code=403, detail="无权限访问该对话")
        
        # 从checkpointer获取对话历史
        global checkpointer
        history = []
        
        if checkpointer:
            try:
                # 使用conversation_id作为session_id来获取状态
                config = {"configurable": {"thread_id": conversation_id}}
                state = await checkpointer.aget_state(config)
                
                if state and state.values and "messages" in state.values:
                    messages = state.values["messages"]
                    for msg in messages:
                        if hasattr(msg, 'type') and hasattr(msg, 'content'):
                            role = "user" if msg.type == "human" else "assistant"
                            history.append({
                                "role": role,
                                "content": msg.content
                            })
            except Exception as e:
                logger.error(f"从checkpointer获取对话历史失败: {str(e)}")
        
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "history": history
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取对话历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取对话历史失败: {str(e)}")

@agent_router.post("/conversations")
async def create_conversation(request: Request):
    """创建新对话"""
    try:
        # 获取当前用户信息
        current_user = get_current_user_from_token(request)
        user_id = current_user.id if current_user else "anonymous"
        
        # 生成新的对话ID
        conversation_id = f"conv_{user_id}_{int(time.time())}"
        
        # 保存到数据库
        # 这里需要实现数据库保存逻辑
        
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "message": "对话创建成功"
        }
    except Exception as e:
        logger.error(f"创建对话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建对话失败: {str(e)}")

@agent_router.post("/conversations/save")
async def save_conversation(request: Request):
    """保存对话到数据库"""
    try:
        # 获取当前用户信息
        current_user = get_current_user_from_token(request)
        user_id = current_user.id if current_user else "anonymous"
        
        # 解析请求数据
        data = await request.json()
        conversation_id = data.get("conversation_id")
        title = data.get("title", "新对话")
        messages = data.get("messages", [])
        
        if not conversation_id:
            raise HTTPException(status_code=400, detail="conversation_id is required")
        
        # 使用checkpointer保存对话到数据库
        global checkpointer
        if checkpointer:
            try:
                # 将消息转换为LangChain消息格式
                from langchain_core.messages import HumanMessage, AIMessage
                langchain_messages = []
                
                for msg in messages:
                    if msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        langchain_messages.append(AIMessage(content=msg["content"]))
                
                # 使用专门的对话历史数据库保存
                try:
                    async with chat_history_pool.connection() as conn:
                        async with conn.cursor() as cur:
                            # 插入或更新对话记录
                            await cur.execute("""
                                INSERT INTO conversations (conversation_id, user_id, title, message_count, updated_at)
                                VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                                ON CONFLICT (conversation_id) 
                                DO UPDATE SET 
                                    title = EXCLUDED.title,
                                    message_count = EXCLUDED.message_count,
                                    updated_at = CURRENT_TIMESTAMP
                            """, (conversation_id, user_id, title, len(messages)))
                            
                            # 删除旧消息
                            await cur.execute("DELETE FROM messages WHERE conversation_id = %s", (conversation_id,))
                            
                            # 插入新消息
                            for msg in messages:
                                await cur.execute("""
                                    INSERT INTO messages (conversation_id, role, content)
                                    VALUES (%s, %s, %s)
                                """, (conversation_id, msg["role"], msg["content"]))
                            
                            await conn.commit()
                            logger.info(f"对话已保存到专用数据库: {conversation_id}, 用户: {user_id}, 消息数: {len(messages)}")
                            
                except Exception as e:
                    logger.error(f"保存到专用数据库失败: {str(e)}")
                    logger.info(f"对话数据已准备: {conversation_id}, 消息数: {len(messages)}")
                
                logger.info(f"对话已保存到数据库: {conversation_id}, 用户: {user_id}, 消息数: {len(messages)}")
                
                return {
                    "status": "success",
                    "conversation_id": conversation_id,
                    "message": "对话保存成功"
                }
            except Exception as e:
                logger.error(f"保存对话到checkpointer失败: {str(e)}")
                raise HTTPException(status_code=500, detail=f"保存对话到数据库失败: {str(e)}")
        else:
            logger.error("checkpointer未初始化")
            raise HTTPException(status_code=503, detail="数据库服务未初始化")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"保存对话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存对话失败: {str(e)}")

@agent_router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """获取特定会话的完整历史"""
    # 检查session_id是否有效
    if not session_id or session_id == "undefined":
        raise HTTPException(status_code=400, detail="无效的会话ID")
        
    try:
        # 从专门的对话历史数据库获取会话历史
        async with chat_history_pool.connection() as conn:
            async with conn.cursor() as cur:
                # 获取会话基本信息
                await cur.execute("""
                    SELECT user_id, title, updated_at, message_count
                    FROM conversations 
                    WHERE conversation_id = %s
                """, (session_id,))
                
                conv_row = await cur.fetchone()
                if not conv_row:
                    return {
                        "session_id": session_id,
                        "conversation_history": [],
                        "last_updated": None
                    }
                
                user_id, title, updated_at, message_count = conv_row
                
                # 获取消息历史
                await cur.execute("""
                    SELECT role, content, created_at
                    FROM messages 
                    WHERE conversation_id = %s 
                    ORDER BY created_at ASC
                """, (session_id,))
                
                message_rows = await cur.fetchall()
                history = []
                for row in message_rows:
                    role, content, created_at = row
                    history.append({
                        "role": role,
                        "content": content
                    })
                
                return {
                    "session_id": session_id,
                    "conversation_history": history,
                    "last_updated": str(updated_at)
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
    try:
        # 从专门的对话历史数据库删除会话
        async with chat_history_pool.connection() as conn:
            async with conn.cursor() as cur:
                # 删除消息（外键约束会自动处理）
                await cur.execute("DELETE FROM messages WHERE conversation_id = %s", (session_id,))
                # 删除会话
                await cur.execute("DELETE FROM conversations WHERE conversation_id = %s", (session_id,))
                await conn.commit()
                
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
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        logger.info("✅ Qdrant客户端初始化成功")
        
        # 2. 初始化数据库连接池（Agent需要）
        logger.info("初始化数据库连接池...")
        # 对话历史专用数据库（从环境变量读取）
        CHAT_HISTORY_DB_URI = os.getenv(
            "CHAT_HISTORY_DB_URI",
            "postgresql://chat_history_user:chat_history_pass@localhost:5432/chat_history_db?sslmode=disable"
        )
        # 原有的LangGraph数据库（从环境变量读取）
        DB_URI = os.getenv(
            "DB_URI",
            "postgresql://postgres:postgres@localhost:5432/langgraph_db?sslmode=disable"
        )
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        # LangGraph数据库连接池
        pool = AsyncConnectionPool(
            conninfo=DB_URI,
            max_size=20,
            kwargs=connection_kwargs
        )
        await pool.open()
        logger.info("初始化数据库检查点存储...")
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        logger.info("✅ LangGraph数据库检查点存储初始化成功")
        
        # 对话历史数据库连接池
        global chat_history_pool
        chat_history_pool = AsyncConnectionPool(
            conninfo=CHAT_HISTORY_DB_URI,
            max_size=10,
            kwargs=connection_kwargs
        )
        await chat_history_pool.open()
        logger.info("✅ 对话历史数据库连接池初始化成功")
        
        # 设置PDF模块的数据库连接池引用
        set_db_pool(pool)
        logger.info("✅ PDF模块数据库连接池引用已设置")
        
        # 初始化用户认证系统
        logger.info("初始化用户认证系统...")
        await create_users_table(pool)
        set_global_pool(pool)
        set_auth_user_manager(pool)
        
        # 初始化USTC OAuth（如果配置了环境变量）
        # 获取应用的基础URL，支持从环境变量读取
        base_url = os.getenv("BASE_URL", "http://localhost:8000")
        init_ustc_oauth(base_url)
        
        logger.info("✅ 用户认证系统初始化成功")
        
        # 从数据库加载用户文档工具
        try:
            await load_user_document_tools_from_db()
            logger.info("✅ 用户文档工具加载完成")
        except Exception as e:
            logger.warning(f"⚠️ 加载用户文档工具失败: {str(e)}")
        
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
# 支持子路径部署
app = FastAPI(
    title="统一知识库与对话Agent服务",
    description="整合知识库管理和对话Agent功能",
    version="1.0.0",
    lifespan=lifespan,
    root_path="/nsrlchat"  # 支持子路径部署
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加文件上传中间件支持
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 添加安全响应头中间件（防止HTTP响应头注入攻击）
app.add_middleware(SecurityHeadersMiddleware)

# 添加认证中间件
app.add_middleware(create_auth_middleware())

# 文档预览API
@kb_router.get("/api/document/{kb_name}/{document_name}/preview")
async def preview_document(kb_name: str, document_name: str):
    """预览文档内容 - 优先使用原文件内容"""
    logger.info(f"预览文档请求: kb_name={kb_name}, document_name={document_name}")
    try:
        # 从Qdrant中获取文档内容
        collection_name = kb_name
        client = QdrantClient(host="localhost", port=6333)
        
        # 搜索包含该文档的点 - 支持部分匹配
        search_result = client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.source",
                        match=qdrant_models.MatchText(text=document_name)  # 使用文本匹配而不是精确匹配
                    )
                ]
            ),
            limit=1  # 只需要一个点就能获取原文件内容
        )
        
        logger.info(f"预览文档查询结果: {len(search_result[0]) if search_result[0] else 0} 个点")
        
        if not search_result[0]:  # 如果没有找到文档
            return {
                "success": False,
                "message": f"文档 '{document_name}' 不存在",
                "data": {}
            }
        
        # 获取第一个点的payload
        first_point = search_result[0][0]
        payload = first_point.payload
        
        # 优先使用原文件内容
        original_content = payload.get("original_content", "")
        source_name = payload.get("source_name", document_name)
        
        if original_content:
            # 使用原文件内容
            logger.info(f"使用原文件内容预览: {source_name}")
            
            # 处理图片路径：将相对路径转换为绝对路径
            import re
            processed_content = original_content
            
            # 查找所有图片引用 ![](image_path)
            def replace_image_path(match):
                image_path = match.group(1)
                # 如果已经是绝对路径，直接返回
                if image_path.startswith('/') or image_path.startswith('http'):
                    return match.group(0)
                
                # 将相对路径转换为绝对路径
                # 格式：![](image_path) -> ![](/marker_outputs/{base_name}/{base_name}_images/{image_path})
                base_name = os.path.splitext(source_name)[0]
                absolute_path = f"/marker_outputs/{base_name}/{base_name}_images/{image_path}"
                return f"![]({absolute_path})"
            
            # 替换所有图片路径
            processed_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image_path, processed_content)
            
            return {
                "success": True,
                "message": f"文档 '{source_name}' 预览内容（原文件）",
                "data": {
                    "document_name": source_name,
                    "content": processed_content,
                    "content_type": "original"
                }
            }
        else:
            # 回退到分块重组方式（兼容旧数据）
            logger.info("原文件内容不存在，使用分块重组方式")
            return await preview_document_fallback(kb_name, document_name, client)
        
    except Exception as e:
        logger.error(f"预览文档失败: {str(e)}")
        return {
            "success": False,
            "message": f"预览文档失败: {str(e)}",
            "data": {}
        }


async def preview_document_fallback(kb_name: str, document_name: str, client):
    """预览文档的备用方法 - 分块重组"""
    try:
        # 搜索包含该文档的所有点 - 支持部分匹配
        search_result = client.scroll(
            collection_name=kb_name,
            scroll_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="metadata.source",
                        match=qdrant_models.MatchText(text=document_name)  # 使用文本匹配而不是精确匹配
                    )
                ]
            ),
            limit=1000
        )
        
        if not search_result[0]:
            return {
                "success": False,
                "message": f"文档 '{document_name}' 不存在",
                "data": {}
            }
        
        # 合并所有块的内容
        chunks = []
        for point in search_result[0]:
            metadata = point.payload.get("metadata", {})
            chunks.append({
                "title": metadata.get("title", ""),
                "content": metadata.get("content", ""),
                "level": metadata.get("level", 1),
                "source": metadata.get("source", "")
            })
        
        # 按level排序，确保正确的层级结构
        chunks.sort(key=lambda x: x.get("level", 1))
        
        # 获取真实的文档名
        real_document_name = document_name
        if chunks:
            real_document_name = chunks[0].get("source", document_name)
        
        # 生成预览内容
        preview_content = ""
        for chunk in chunks:
            if chunk["title"]:
                level = chunk.get("level", 1)
                preview_content += f"{'#' * level} {chunk['title']}\n\n"
            if chunk["content"]:
                preview_content += chunk["content"] + "\n\n"
        
        # 处理图片路径：将相对路径转换为绝对路径
        import re
        def replace_image_path(match):
            image_path = match.group(1)
            # 如果已经是绝对路径，直接返回
            if image_path.startswith('/') or image_path.startswith('http'):
                return match.group(0)
            
            # 将相对路径转换为绝对路径
            # 格式：![](image_path) -> ![](/marker_outputs/{base_name}/{base_name}_images/{image_path})
            base_name = os.path.splitext(real_document_name)[0]
            absolute_path = f"/marker_outputs/{base_name}/{base_name}_images/{image_path}"
            return f"![]({absolute_path})"
        
        # 替换所有图片路径
        preview_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_image_path, preview_content)
        
        return {
            "success": True,
            "message": f"文档 '{real_document_name}' 预览内容（分块重组）",
            "data": {
                "document_name": real_document_name,
                "content": preview_content,
                "content_type": "reconstructed",
                "chunks_count": len(chunks)
            }
        }
        
    except Exception as e:
        logger.error(f"分块重组预览失败: {str(e)}")
        return {
            "success": False,
            "message": f"分块重组预览失败: {str(e)}",
            "data": {}
        }

# 注册子应用
# 添加静态文件服务（必须在路由之前）
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
print(f"静态文件目录: {static_dir}")
print(f"ustc.svg存在: {os.path.exists(os.path.join(static_dir, 'ustc.svg'))}")

# 使用绝对路径挂载静态文件
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 添加一个简单的测试路由来验证静态文件
@app.get("/test-static")
async def test_static():
    return {"static_dir": static_dir, "exists": os.path.exists(os.path.join(static_dir, "ustc.svg"))}

# 添加一个直接的路由来服务SVG文件
@app.get("/ustc.svg")
async def serve_ustc_svg():
    svg_path = os.path.join(static_dir, "ustc.svg")
    if os.path.exists(svg_path):
        return FileResponse(svg_path, media_type="image/svg+xml")
    else:
        raise HTTPException(status_code=404, detail="SVG file not found")

# 添加子路径版本的SVG路由
@app.get("/nsrlchat/ustc.svg")
async def serve_ustc_svg_subpath():
    svg_path = os.path.join(static_dir, "ustc.svg")
    if os.path.exists(svg_path):
        return FileResponse(svg_path, media_type="image/svg+xml")
    else:
        raise HTTPException(status_code=404, detail="SVG file not found")

# 添加marker输出图片的静态文件服务
app.mount("/marker_outputs", StaticFiles(directory="marker_outputs"), name="marker_outputs")

app.include_router(kb_router)
app.include_router(agent_router)
app.include_router(auth_router)

# 添加根路径重定向到Web界面（需要认证）
@app.get("/")
async def read_root(current_user: UserResponse = Depends(get_current_user)):
    return FileResponse('ustc/static/index.html')

# 欢迎页面路由（暂时保留，等USTC申请完成后启用）
# @app.get("/welcome")
# async def welcome_page():
#     """显示欢迎页面"""
#     return FileResponse('ustc/static/welcome.html')

# 添加上传页面路由（需要知识库贡献者或管理员权限）
@app.get("/upload.html")
async def upload_page(current_user: UserResponse = Depends(get_current_contributor_user)):
    return FileResponse('ustc/static/upload.html')

# 添加测试页面路由
@app.get("/test_documents.html")
async def test_documents_page():
    return FileResponse('test_documents.html')

# 添加上传测试页面路由
@app.get("/test_upload.html")
async def test_upload_page():
    return FileResponse('../test_upload.html')


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