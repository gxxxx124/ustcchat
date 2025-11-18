# user_document_processor.py
import os
import re
import uuid
import requests
import logging
import time
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.tools import Tool
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.models import ScoredPoint
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from pydantic import BaseModel
import json

# 配置日志
logger = logging.getLogger("user_document_processor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# 添加输入模式类（与RAG工具保持一致）
class DocumentSearchInput(BaseModel):
    query: str  # 仅接受查询字符串


def convert_to_langchain_docs(chunks):
    """将结构化块转换为LangChain文档格式"""
    docs = []
    for chunk in chunks:
        doc_id = f"{chunk['source']}_chunk_{hash(chunk['content_text'][:100])}"
        docs.append(
            Document(
                page_content=chunk["content_text"],
                metadata={
                    "title": chunk["title_text"],
                    "content": chunk["content_text"],
                    "level": chunk["level"],
                    "parent_title": chunk["parent_title"],
                    "path": chunk["path"],
                    "source": chunk["source"],
                    "id": doc_id,
                    **{k: v for k, v in chunk.items() if k not in ["title_text", "content_text"]}
                }
            )
        )
    return docs


def parse_pdf_file(file_path: str, max_pages: int = 50) -> List[Dict]:
    """
    解析PDF文件并转换为结构化块

    参数:
    - file_path: PDF文件路径
    - max_pages: 最大处理页数（防止大文件）

    返回:
    - 结构化块列表
    """
    logger.info(f"�� 正在解析PDF文件: {file_path}")
    chunks = []
    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)

        # 限制处理页数
        pages_to_process = min(total_pages, max_pages)
        logger.info(f"�� 检测到 {total_pages} 页，将处理前 {pages_to_process} 页")

        # 提取PDF元数据
        metadata = reader.metadata
        if metadata is not None:
            title = metadata.get('/Title', os.path.basename(file_path))
        else:
            title = os.path.basename(file_path)

        # 文本分块器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )

        # 处理每一页
        for page_num in range(pages_to_process):
            page = reader.pages[page_num]
            text = page.extract_text()

            if not text or len(text.strip()) == 0:
                continue

            # 分块处理
            page_chunks = text_splitter.split_text(text)

            # 构建结构化块
            for i, chunk_text in enumerate(page_chunks):
                chunks.append({
                    "title_text": f"第{page_num + 1}页",
                    "content_text": chunk_text,
                    "level": 1,
                    "parent_title": title,
                    "path": f"{title}/第{page_num + 1}页",
                    "source": os.path.basename(file_path),
                    "page": page_num + 1,
                    "chunk_index": i
                })

        logger.info(f"✅ 成功解析PDF文件，生成 {len(chunks)} 个文本块")
        import json
        with open('/home/easyai/test.json', 'w', encoding='utf-8') as json_file:
            json.dump(chunks, json_file, ensure_ascii=False, indent=2)

        return chunks

    except Exception as e:
        logger.error(f"❌ PDF解析失败: {str(e)}")
        raise


class EnhancedQdrantVectorStore:
    """完全自定义的向量存储类，适配Qdrant服务器模式"""

    def __init__(
            self,
            client: Any,
            collection_name: str,
            embedding: Any
    ):
        self.client = client
        self.collection_name = collection_name
        self.embedding_model = embedding

    def create_collection_if_not_exists(self, vector_size: int = 1024):
        """创建支持多向量的集合（如果不存在）"""
        try:
            # 检查集合是否存在
            self.client.get_collection(self.collection_name)
            logger.info(f"✅ 集合 {self.collection_name} 已存在，跳过创建")
        except Exception:
            logger.info(f"�� 创建集合 {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "title": VectorParams(size=vector_size, distance=Distance.COSINE),
                    "content": VectorParams(size=vector_size, distance=Distance.COSINE)
                }
            )
            logger.info(f"✅ 集合 {self.collection_name} 创建成功")

    def weighted_hybrid_search(
            self,
            query: str,
            k: int = 5,
            title_weight: float = 0.2,
            content_weight: float = 0.8
    ) -> List[Tuple[Document, float]]:
        """加权融合搜索标题和内容（兼容Qdrant 1.7.0+）"""
        try:
            # 获取查询向量
            query_vector = self.embedding_model.embed_query(query)

            # 分别搜索标题和内容 - 使用新版API格式
            title_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("title", query_vector),  # ✅ 新版API格式
                query_filter=None,
                with_payload=True,
                with_vectors=False,
                limit=k * 3
            )

            content_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("content", query_vector),  # ✅ 新版API格式
                query_filter=None,
                with_payload=True,
                with_vectors=False,
                limit=k * 3
            )

            # 融合结果
            combined_scores = {}
            for result in title_results:
                doc_id = result.payload["metadata"]["id"]
                score = result.score * title_weight
                combined_scores[doc_id] = (score, result)

            for result in content_results:
                doc_id = result.payload["metadata"]["id"]
                score = result.score * content_weight
                if doc_id in combined_scores:
                    combined_scores[doc_id] = (
                        combined_scores[doc_id][0] + score,
                        combined_scores[doc_id][1]
                    )
                else:
                    combined_scores[doc_id] = (score, result)

            # 排序并返回前k个结果
            sorted_results = sorted(
                combined_scores.values(),
                key=lambda x: x[0],
                reverse=True
            )[:k]

            # 转换为Document格式
            return [
                (
                    self._document_from_scored_point(result[1]),
                    result[0]
                )
                for result in sorted_results
            ]
        except Exception as e:
            logger.error(f"Qdrant搜索出错: {str(e)}")
            raise

    def _document_from_scored_point(self, scored_point: ScoredPoint) -> Document:
        payload = scored_point.payload
        metadata = payload.get("metadata", {})
        return Document(
            page_content=payload.get("page_content", ""),
            metadata=metadata
        )

    def delete(self, filter: Optional[Filter] = None) -> None:
        """删除满足条件的点"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=filter
        )


class UserDocumentVectorStore(EnhancedQdrantVectorStore):
    """专为用户上传文档设计的向量存储类"""

    def __init__(
            self,
            client: Any,
            collection_name: str,
            embedding: Any,
            document_id: str,
            document_name: str,
            user_id: Optional[str] = None
    ):
        super().__init__(client, collection_name, embedding)
        self.document_id = document_id
        self.document_name = document_name
        self.user_id = user_id

    def create_collection_if_not_exists(self, vector_size: int = 1024):
        """创建用户文档专用集合（如果不存在）"""
        try:
            # 检查集合是否存在
            self.client.get_collection(self.collection_name)
            logger.info(f"✅ 用户文档集合 {self.collection_name} 已存在，跳过创建")
        except Exception:
            logger.info(f"�� 创建用户文档集合 {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "title": VectorParams(size=vector_size, distance=Distance.COSINE),
                    "content": VectorParams(size=vector_size, distance=Distance.COSINE)
                }
            )
            logger.info(f"✅ 用户文档集合 {self.collection_name} 创建成功")

    def upsert_document_chunks(self, chunks: List[Dict]) -> Any:
        """上传文档块到向量数据库"""
        docs = convert_to_langchain_docs(chunks)

        # 准备多向量数据点
        points = []
        for i, doc in enumerate(docs):
            # 添加用户文档特定元数据
            doc.metadata["document_id"] = self.document_id
            doc.metadata["document_name"] = self.document_name
            if self.user_id:
                doc.metadata["user_id"] = self.user_id

            title_vector = self.embedding_model.embed_query(
                f"标题: {doc.metadata['path']}"
            )
            content_vector = self.embedding_model.embed_query(
                f"内容: {doc.metadata['content']}"
            )

            # 生成唯一ID
            point_id = hash(f"{self.document_id}_{i}") % (2 ** 63)

            points.append(PointStruct(
                id=point_id,
                vector={
                    "title": title_vector,
                    "content": content_vector
                },
                payload={
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
            ))

        # 批量上传
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )

        logger.info(f"✅ 已上传 {len(points)} 个用户文档块到集合 {self.collection_name}")
        return operation_info

    def create_search_tool(self):
        """创建针对此文档的搜索工具，返回LangChain Tool对象"""

        def search_tool_func(query: str) -> str:
            """在特定用户文档中搜索相关内容"""
            try:
                # 执行搜索，过滤特定文档
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=("content", self.embedding_model.embed_query(query)),
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.document_id",
                                match=MatchValue(value=self.document_id)
                            )
                        ]
                    ),
                    limit=3
                )

                # 构建响应
                if not results:
                    return "未在您的文档中找到相关信息。"

                response = f"【基于您的基因检测报告 '{self.document_name}' 的查询结果】\n\n"
                highest_similarity = 0.0

                for i, result in enumerate(results):
                    similarity = result.score
                    highest_similarity = max(highest_similarity, similarity)
                    content = result.payload.get("page_content", "内容不可用")

                    response += f"结果 #{i + 1} (相似度: {similarity:.4f})\n"
                    response += f"内容: {content[:300]}...\n"
                    response += "----------------------------------------\n"

                # 添加最高相似度信息（用于Agent决策逻辑）
                response += f"最高相似度: {highest_similarity:.4f}\n"

                return response

            except Exception as e:
                logger.error(f"❌ 搜索用户文档时发生错误: {str(e)}")
                return f"❌ 搜索过程中发生错误: {str(e)}"

        # 使用Tool.from_function创建工具（与RAG工具相同格式）
        return Tool.from_function(
            name=f"search_{self.document_id}",
            description=f"搜索用户上传的基因检测报告 '{self.document_name}'。仅需要输入查询语句。",
            func=search_tool_func,
            args_schema=DocumentSearchInput,
            return_direct=False
        )


def process_document_from_url(
        url: str,
        document_id: str,  # 现在是必填参数，且等于session_id
        document_name: Optional[str] = None,
        user_id: Optional[str] = None,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "user_documents"
) -> Dict[str, Any]:
    """
    从URL处理文档并创建搜索工具

    参数:
    - url: 文档URL
    - document_id: 文档ID（与session_id相同）
    - document_name: 可选，自定义文档名称
    - user_id: 可选，关联的用户ID
    - host: Qdrant主机
    - port: Qdrant端口
    - collection_name: 集合名称

    返回:
    - 包含搜索工具信息的字典
    """
    # document_id现在是必填参数，不需要生成
    if not document_name:
        # 从URL提取文件名
        document_name = url.split("/")[-1].split("?")[0].split("#")[0]
        if not document_name or "." not in document_name:
            document_name = f"session_{document_id}_document"

    logger.info(f"�� 开始处理文档: {document_name} (ID: {document_id})")

    try:
        # 1. 下载文档
        local_dir = "/home/easyai/yonghu"
        oss_path = url.split("/", 3)[-1]
        pdfname = f'{document_name}.pdf'
        local_input = os.path.join(local_dir, pdfname)
        logger.info(f"⬇️ 正在下载文档: {url}")

        # 下载OSS文件（这部分可以保留，但需要确保oss2已安装）
        try:
            from oss2 import Auth, Bucket
            # OSS 配置（从环境变量读取）
            OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID", "")
            OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET", "")
            OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "https://oss-cn-hangzhou.aliyuncs.com")
            OSS_BUCKET = os.getenv("OSS_BUCKET", "")

            # 初始化 OSS 客户端
            auth = Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
            bucket = Bucket(auth, OSS_ENDPOINT, OSS_BUCKET)

            # 下载文件
            bucket.get_object_to_file(url, local_input)
            bucket.get_object_to_file(url, '/home/easyai/test.pdf')
            logger.info(f"✅ OSS文件下载成功: {local_input}")
        except ImportError:
            logger.warning("⚠️ oss2模块未安装，尝试使用requests下载")
            # 如果oss2未安装，尝试使用requests
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(local_input, "wb") as f:
                f.write(response.content)
            logger.info(f"✅ 使用requests下载成功: {local_input}")

        # 2. 解析文档（根据文件类型）
        if local_input.lower().endswith('.pdf'):
            logger.info("�� 检测到PDF文件，开始解析...")
            chunks = parse_pdf_file(local_input)
        elif local_input.lower().endswith(('.md', '.markdown')):
            logger.info("�� 检测到Markdown文件，开始解析...")
            try:
                from md2chunks import parse_markdown_file
                chunks = parse_markdown_file(local_input)
            except ImportError:
                logger.error("❌ 无法导入md2chunks模块")
                raise
        else:
            # 尝试作为文本文件处理
            logger.info("�� 未知文件类型，尝试作为文本文件处理...")
            with open(local_input, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            chunks = [{
                "title_text": document_name,
                "content_text": text[:5000],  # 限制文本长度
                "level": 1,
                "parent_title": document_name,
                "path": document_name,
                "source": document_name,
                "page": 1,
                "chunk_index": 0
            }]

        # 3. 初始化向量存储
        logger.info(f"�� 初始化向量存储: {collection_name}")
        client = QdrantClient(host=host, port=port)

        # 4. 创建用户文档向量存储
        # 假设您有QwenEmbedding类可用
        try:
            from embedding import QwenEmbedding
            embedding_model = QwenEmbedding()
        except ImportError:
            # 如果QwenEmbedding不可用，使用替代方案
            from langchain_openai import OpenAIEmbeddings
            embedding_model = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_base="http://localhost:11434/v1",
                openai_api_key="none"
            )

        vector_store = UserDocumentVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_model,
            document_id=document_id,
            document_name=document_name,
            user_id=user_id
        )

        # 5. 确保集合存在
        vector_store.create_collection_if_not_exists()

        # 6. 上传文档块
        logger.info("⬆️ 正在上传文档块到向量数据库...")
        vector_store.upsert_document_chunks(chunks)

        # 7. 创建搜索工具
        logger.info("��️ 创建搜索工具...")
        search_tool = vector_store.create_search_tool()

        # 8. 清理临时文件
        logger.info("�� 清理临时文件...")
        os.remove(local_input)

        logger.info(f"�� 文档处理完成! 可使用工具 '{search_tool.name}' 进行搜索")

        return {
            "status": "success",
            "document_id": document_id,
            "document_name": document_name,
            "tool": search_tool  # 返回Tool对象而不是字典
        }

    except Exception as e:
        logger.exception(f"❌ 文档处理失败: {str(e)}")
        # 清理临时文件（如果存在）
        if 'local_input' in locals() and os.path.exists(local_input):
            try:
                os.remove(local_input)
                logger.info(f"�� 已清理失败任务的临时文件: {local_input}")
            except Exception as cleanup_error:
                logger.error(f"�� 清理临时文件失败: {str(cleanup_error)}")
        raise


# 全局存储用户文档工具
user_document_tools = {}  # 用于存储用户文档工具

# 添加数据库连接池引用
db_pool = None

def set_db_pool(pool):
    """设置数据库连接池引用"""
    global db_pool
    db_pool = pool

async def save_user_document_tools_to_db():
    """将用户文档工具信息保存到数据库"""
    global db_pool, user_document_tools
    if not db_pool:
        logger.warning("数据库连接池未设置，无法保存工具信息")
        return False
    
    try:
        async with db_pool.connection() as conn:
            # 创建工具存储表（如果不存在）
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_document_tools (
                    tool_name VARCHAR(255) PRIMARY KEY,
                    document_id VARCHAR(255) NOT NULL,
                    document_name VARCHAR(255),
                    user_id VARCHAR(255),
                    tool_config JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 清空现有数据
            await conn.execute("DELETE FROM user_document_tools")
            
            # 插入工具信息
            for tool_name, tool_info in user_document_tools.items():
                tool_config = {
                    "name": tool_info["tool"].name,
                    "description": tool_info["tool"].description,
                    "args_schema": str(tool_info["tool"].args_schema) if hasattr(tool_info["tool"], "args_schema") else "{}"
                }
                
                await conn.execute("""
                    INSERT INTO user_document_tools (tool_name, document_id, document_name, user_id, tool_config)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    tool_name,
                    tool_info["document_id"],
                    tool_info["document_name"],
                    tool_info["user_id"],
                    json.dumps(tool_config)
                ))
            
            await conn.commit()
            logger.info(f"✅ 成功保存 {len(user_document_tools)} 个用户文档工具到数据库")
            return True
            
    except Exception as e:
        logger.error(f"❌ 保存用户文档工具到数据库失败: {str(e)}")
        return False

async def load_user_document_tools_from_db():
    """从数据库加载用户文档工具信息"""
    global db_pool, user_document_tools
    if not db_pool:
        logger.warning("数据库连接池未设置，无法加载工具信息")
        return False
    
    try:
        async with db_pool.connection() as conn:
            # 检查表是否存在
            result = await conn.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'user_document_tools'
                )
            """)
            table_exists = await result.fetchone()
            
            if not table_exists[0]:
                logger.info("用户文档工具表不存在，跳过加载")
                return True
            
            # 查询所有工具信息
            result = await conn.execute("""
                SELECT tool_name, document_id, document_name, user_id, tool_config
                FROM user_document_tools
                ORDER BY created_at DESC
            """)
            
            rows = await result.fetchall()
            loaded_count = 0
            
            for row in rows:
                tool_name, document_id, document_name, user_id, tool_config = row
                
                try:
                    # 重新创建工具对象
                    tool = create_user_document_search_tool(
                        document_id=document_id,
                        document_name=document_name or "用户上传的文档"
                    )
                    
                    # 存储到全局变量
                    user_document_tools[tool_name] = {
                        "tool": tool,
                        "document_id": document_id,
                        "document_name": document_name,
                        "user_id": user_id
                    }
                    
                    loaded_count += 1
                    logger.info(f"✅ 成功加载工具: {tool_name} - {document_name}")
                    
                except Exception as e:
                    logger.error(f"❌ 加载工具 {tool_name} 失败: {str(e)}")
                    continue
            
            logger.info(f"✅ 成功从数据库加载 {loaded_count} 个用户文档工具")
            return True
            
    except Exception as e:
        logger.error(f"❌ 从数据库加载用户文档工具失败: {str(e)}")
        return False

def create_user_document_search_tool(document_id: str, document_name: str):
    """创建用户文档搜索工具"""
    from langchain.tools import Tool
    
    def search_user_document(query: str) -> str:
        """搜索用户上传的文档"""
        try:
            # 这里应该实现实际的文档搜索逻辑
            # 暂时返回占位符信息
            return f"基于您的文档 '{document_name}' 的查询结果: {query}\n\n注意：这是重新加载的工具，需要重新处理文档内容。"
        except Exception as e:
            return f"搜索失败: {str(e)}"
    
    return Tool(
        name=f"search_{document_id}",
        description=f"搜索用户上传的文档 '{document_name}'。仅需要输入查询语句。",
        func=search_user_document
    )


async def register_user_document_tool(
        url: str,
        document_id: str,  # 与session_id相同
        user_id: Optional[str] = None,
        document_name: Optional[str] = None
) -> str:
    """
    注册用户上传的文档搜索工具

    参数:
    - url: 文档URL
    - document_id: 文档ID（与session_id相同）
    - user_id: 可选，关联的用户ID
    - document_name: 可选，自定义文档名称

    返回:
    - 工具名称
    """
    try:
        logger.info(f"�� 注册用户文档工具，URL: {url}, 用户ID: {user_id}, 文档ID: {document_id}")

        # 处理文档并获取工具信息
        result = process_document_from_url(
            url=url,
            document_id=document_id,
            document_name=document_name,
            user_id=user_id
        )

        # 存储工具（现在直接存储Tool对象）
        tool = result["tool"]
        tool_name = tool.name
        user_document_tools[tool_name] = {
            "tool": tool,
            "document_id": result["document_id"],
            "document_name": result["document_name"],
            "user_id": user_id
        }

        logger.info(f"✅ 成功注册用户文档工具: {tool_name}")
        
        # 自动保存到数据库
        if db_pool:
            try:
                await save_user_document_tools_to_db()
                logger.info(f"✅ 工具信息已自动保存到数据库")
            except Exception as e:
                logger.warning(f"⚠️ 自动保存工具信息到数据库失败: {str(e)}")
        else:
            logger.warning(f"⚠️ 数据库连接池未设置，无法自动保存工具信息")
        
        return tool_name

    except Exception as e:
        logger.error(f"❌ 注册用户文档工具失败: {str(e)}")
        raise


def get_user_document_tool(tool_name: str) -> Optional[Dict]:
    """获取指定名称的用户文档工具"""
    return user_document_tools.get(tool_name)


def get_user_document_tool_by_session(session_id: str) -> Optional[Dict]:
    """根据session_id获取用户文档工具"""
    tool_name = f"search_{session_id}"
    return user_document_tools.get(tool_name)


def list_user_document_tools(user_id: Optional[str] = None) -> List[Dict]:
    """列出所有用户文档工具，可按用户ID过滤"""
    tools = []
    for tool_name, tool_info in user_document_tools.items():
        if user_id is None or tool_info.get("user_id") == user_id:
            tools.append({
                "tool_name": tool_name,
                "document_id": tool_info["document_id"],
                "document_name": tool_info["document_name"],
                "description": tool_info["tool"].description
            })
    return tools


def cleanup_temp_files():
    """清理临时文档文件"""
    temp_dir = "/home/easyai/yonghu"
    if os.path.exists(temp_dir):
        cutoff_time = time.time() - 24 * 3600  # 24小时前
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            try:
                if os.path.getmtime(file_path) < cutoff_time:
                    os.remove(file_path)
                    logger.info(f"�� 已清理过期临时文件: {file}")
            except Exception as e:
                logger.error(f"�� 清理临时文件失败: {file} - {str(e)}")
        logger.info("�� 临时文件清理完成")