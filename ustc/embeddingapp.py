from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import os
import requests
import tempfile
import shutil
import uuid
from chunks2embedding import (
    embedding_init,
    upsert_md_file,
    delete_by_source,
    list_all_collections,
    get_collection_info
)
import logging
import oss2
from pdf2md import pdf2md
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, Filter, FieldCondition, MatchValue



# 初始化 OSS 客户端
auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET)

# 本地临时路径
LOCAL_DIR = "/home/easyai/oss"
os.makedirs(LOCAL_DIR, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("embeddingapp")


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



# 2. 修复KnowledgeBaseResponse定义 - 关键修复在这里
class KnowledgeBaseResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None  # ✅ 修复：添加了字段名"data"


# 3. 创建FastAPI应用
app = FastAPI(
    title="知识库管理API",
    description="修复类型定义错误的API服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 4. 简化的关键函数：只获取当前知识库信息
def get_current_knowledge_base_info(kb_name: str):
    """只获取指定知识库的文档信息（不包含其他知识库）"""
    try:
        # 创建Qdrant客户端
        client = QdrantClient(host="localhost", port=6333)

        # 检查集合是否存在
        try:
            client.get_collection(kb_name)
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
            points, next_offset = client.scroll(
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


# 5. 修复文档下载函数
def download_pdf_from_oss(file_key, local_path):
    try:
        bucket.get_object_to_file(file_key, local_path)
        logger.info(f"文件下载成功: {local_path}")
        return True
    except Exception as e:
        logger.error(f"文件下载失败: {str(e)}")
        return False


# 6. 简化后的API端点
@app.post("/api/knowledge-base", response_model=KnowledgeBaseResponse)
async def manage_knowledge_base(request: KnowledgeBaseRequest):
    """统一的知识库管理端点，只返回当前知识库信息"""
    try:
        if request.action == KnowledgeBaseAction.CREATE:
            # 检查集合是否已存在
            client = QdrantClient(host="localhost", port=6333)

            try:
                # 尝试获取集合，如果存在则返回exists
                client.get_collection(request.name)
                status = "exists"
                message = f"知识库 '{request.name}' 已存在"
            except:
                # 集合不存在，创建它
                try:
                    client.create_collection(
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
                # 检查集合是否存在
                client = QdrantClient(host="localhost", port=6333)
                client.delete_collection(request.name)
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
            if not request.document_name :
                return KnowledgeBaseResponse(
                    success=False,
                    message="上传文档需要提供 document_name" ,
                    data={"name": request.name}
                )

            # 下载文件到临时目录
            oss_path =f"knowledge-documents/{request.name}/{request.document_name}.pdf"
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

                # 下载文件
                pdf2md(local_input)
                tempfile = f'/home/easyai/OCRFlux/localworkspace/markdowns/{request.document_name}/{request.document_name}.md'
                # 直接调用您已有的函数
                vector_store = embedding_init(collection_name=request.name)
                operation_info = upsert_md_file(tempfile, vector_store)

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

        elif request.action == KnowledgeBaseAction.DELETE_DOCUMENT:
            if not request.document_name:
                return KnowledgeBaseResponse(
                    success=False,
                    message="删除文档需要提供 document_name",
                    data={"name": request.name}
                )

            # 直接调用您已有的函数
            deletename = f'{request.document_name}.md'
            vector_store = embedding_init(collection_name=request.name)
            operation_info = delete_by_source(deletename, vector_store)

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
@app.get("/api/knowledge-bases", response_model=KnowledgeBaseResponse)
async def list_knowledge_bases():
    """列出所有知识库 - 保持不变，因为这是另一个功能"""
    try:
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections().collections

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


@app.get("/api/knowledge-base/{kb_name}/documents", response_model=KnowledgeBaseResponse)
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


@app.get("/health", response_model=KnowledgeBaseResponse)
async def health_check():
    """健康检查"""
    try:
        client = QdrantClient(host="localhost", port=6333)
        client.get_collections()
        return KnowledgeBaseResponse(
            success=True,
            message="服务运行正常",
            data={"port": 5001}
        )
    except Exception as e:
        return KnowledgeBaseResponse(
            success=False,
            message=f"服务异常: {str(e)}",
            data={"port": 5001}
        )


if __name__ == "__main__":
    import uvicorn

    logger.info("�� 启动知识库管理API服务...")
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")