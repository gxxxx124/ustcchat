from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain_core.documents import Document
from embedding import QwenEmbedding
import os
import logging
import gc
import torch
from md2chunks import parse_markdown_file, parse_markdown_file_api
from typing import List, Dict, Any, Tuple, Optional
from qdrant_client.models import ScoredPoint

# 配置日志
logger = logging.getLogger(__name__)

# 全局单例：embedding 模型只初始化一次
_global_embedding_model = None


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
            # 检查集合是否存在 - 使用collections_list API
            collections = self.client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if self.collection_name in existing_collections:
                print(f"✅ 集合 {self.collection_name} 已存在，跳过创建")
                return
        except Exception as e:
            print(f"⚠️ 检查集合时出错: {e}")
        
        # 创建集合
        try:
            print(f"🔄 创建集合 {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "title": VectorParams(size=vector_size, distance=Distance.COSINE),
                    "content": VectorParams(size=vector_size, distance=Distance.COSINE)
                }
            )
            print(f"✅ 集合 {self.collection_name} 创建成功")
        except Exception as e:
            print(f"❌ 创建集合失败: {e}")
            # 如果集合已存在，忽略错误
            if "already exists" in str(e).lower():
                print(f"✅ 集合 {self.collection_name} 已存在")
            else:
                raise e

    def weighted_hybrid_search(
            self,
            query: str,
            k: int = 5,
            title_weight: float = 0.7,
            content_weight: float = 0.3
    ) -> List[Tuple[Document, float]]:
        """加权融合搜索标题和内容（兼容Qdrant 1.7.0+）"""
        try:
            # 获取查询向量
            query_vector = self.embedding_model.embed_query(query)

            # 分别搜索标题和内容 - 使用新版API格式
            # 增加limit到k*10，确保包含关键词的结果能被包含（即使排名靠后）
            title_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("title", query_vector),  # ✅ 新版API格式
                query_filter=None,
                with_payload=True,
                with_vectors=False,
                limit=k * 10  # 从k*3增加到k*10，确保能包含更多相关结果
            )

            content_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("content", query_vector),  # ✅ 新版API格式
                query_filter=None,
                with_payload=True,
                with_vectors=False,
                limit=k * 10  # 从k*3增加到k*10，确保能包含更多相关结果
            )
            
            # 提取查询关键词（用于关键词匹配加分）
            query_keywords = set(query.lower().split())

            # 合并结果并计算加权分数
            combined_results = {}
            
            # 处理标题搜索结果
            for result in title_results:
                doc_id = result.id
                payload = result.payload
                metadata = payload.get('metadata', {})
                title = metadata.get('title', metadata.get('title_text', '')).lower()
                
                # 计算基础title分数
                base_title_score = result.score * title_weight
                
                # 关键词匹配加分：如果title包含查询关键词，进行加分
                title_contains_keyword = any(keyword in title for keyword in query_keywords if len(keyword) > 1)
                if title_contains_keyword:
                    # 对包含关键词的title进行加分（乘以1.5），确保包含关键词的标题排名更靠前
                    base_title_score *= 1.5
                
                if doc_id not in combined_results:
                    combined_results[doc_id] = {
                        'title_score': base_title_score,
                        'content_score': 0.0,
                        'payload': payload,
                        'total_score': base_title_score
                    }
                else:
                    combined_results[doc_id]['title_score'] = base_title_score
                    combined_results[doc_id]['total_score'] += base_title_score

            # 处理内容搜索结果
            for result in content_results:
                doc_id = result.id
                payload = result.payload
                metadata = payload.get('metadata', {})
                title = metadata.get('title', metadata.get('title_text', '')).lower()
                
                # 计算基础content分数
                base_content_score = result.score * content_weight
                
                # 关键词匹配加分：如果title包含查询关键词，对content分数也进行小幅加分
                title_contains_keyword = any(keyword in title for keyword in query_keywords if len(keyword) > 1)
                if title_contains_keyword:
                    # 对包含关键词的content进行小幅加分（乘以1.2）
                    base_content_score *= 1.2
                
                if doc_id not in combined_results:
                    combined_results[doc_id] = {
                        'title_score': 0.0,
                        'content_score': base_content_score,
                        'payload': payload,
                        'total_score': base_content_score
                    }
                else:
                    combined_results[doc_id]['content_score'] = base_content_score
                    combined_results[doc_id]['total_score'] += base_content_score

            # 为QA对类型的内容增加Q的权重
            for doc_id, result in combined_results.items():
                payload = result['payload']
                # 检查是否为QA对
                if payload.get('metadata', {}).get('type') == 'qa':
                    # QA对中问题权重增加
                    result['total_score'] *= 1.5  # 增加50%的权重
                    # 如果标题分数较高（问题匹配），进一步增加权重
                    if result['title_score'] > result['content_score']:
                        result['total_score'] *= 1.3  # 问题匹配再增加30%权重

            # 按总分排序并返回前k个结果
            sorted_results = sorted(
                combined_results.items(),
                key=lambda x: x[1]['total_score'],
                reverse=True
            )[:k]

            # 转换为Document对象
            documents = []
            for doc_id, result in sorted_results:
                payload = result['payload']
                # 对于QA对，返回完整的问答内容
                if payload.get('metadata', {}).get('type') == 'qa':
                    # 构建完整的QA对内容
                    question = payload.get('metadata', {}).get('question', '')
                    answer = payload.get('metadata', {}).get('answer', '')
                    qa_content = f"问题：{question}\n\n答案：{answer}"
                    
                    # 创建Document对象，标记为QA对
                    doc = Document(
                        page_content=qa_content,
                        metadata={
                            **payload.get('metadata', {}),
                            'is_qa_pair': True,  # 标记这是QA对
                            'source_type': 'qa_knowledge_base'  # 标记来源类型
                        }
                    )
                else:
                    # 普通文档
                    doc = Document(
                        page_content=payload.get('page_content', ''),
                        metadata=payload.get('metadata', {})
                    )
                
                documents.append((doc, result['total_score']))

            return documents

        except Exception as e:
            logger.error(f"加权融合搜索失败: {e}")
            # 降级到简单向量搜索
            return self._simple_vector_search(query, k)

    def _simple_vector_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """简单的向量搜索作为降级方案"""
        try:
            # 生成查询向量
            query_vector = self.embedding.embed_query(query)
            
            # 执行向量搜索
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k
            )
            
            # 转换为Document对象
            documents = []
            for result in search_results:
                payload = result.payload
                metadata = payload.get("metadata", {})
                
                # 对于QA对，返回完整的问答内容
                if metadata.get('type') == 'qa':
                    question = metadata.get('question', '')
                    answer = metadata.get('answer', '')
                    qa_content = f"问题：{question}\n\n答案：{answer}"
                    
                    doc = Document(
                        page_content=qa_content,
                        metadata={
                            **metadata,
                            'is_qa_pair': True,
                            'source_type': 'qa_knowledge_base'
                        }
                    )
                else:
                    doc = Document(
                        page_content=payload.get('page_content', ''),
                        metadata=metadata
                    )
                
                documents.append((doc, result.score))
            
            return documents
            
        except Exception as e:
            logger.error(f"简单向量搜索也失败: {e}")
            return []

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


def convert_to_langchain_docs(chunks):
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
                    "id": doc_id
                }
            )
        )
    return docs


def embedding_init(
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "nsrl_tech_docs"
):
    """
    初始化Qdrant客户端（服务器模式）

    参数:
    - host: Qdrant服务器主机地址
    - port: Qdrant服务器端口
    - collection_name: 集合名称
    """
    # 连接到Qdrant服务器
    if host.startswith(('http://', 'https://')):
        # 从URL中提取主机名
        host = host.split('://')[1].split(':')[0]

    # 临时禁用代理，避免对本地连接造成影响
    old_http_proxy = os.environ.pop('http_proxy', None)
    old_https_proxy = os.environ.pop('https_proxy', None)
    old_HTTP_PROXY = os.environ.pop('HTTP_PROXY', None)
    old_HTTPS_PROXY = os.environ.pop('HTTPS_PROXY', None)

    try:
        client = QdrantClient(host=host, port=port)
    finally:
        # 恢复代理环境变量
        if old_http_proxy:
            os.environ['http_proxy'] = old_http_proxy
        if old_https_proxy:
            os.environ['https_proxy'] = old_https_proxy
        if old_HTTP_PROXY:
            os.environ['HTTP_PROXY'] = old_HTTP_PROXY
        if old_HTTPS_PROXY:
            os.environ['HTTPS_PROXY'] = old_HTTPS_PROXY

    # 使用单例模式：embedding 模型只初始化一次（全局共享）
    global _global_embedding_model
    if '_global_embedding_model' not in globals() or _global_embedding_model is None:
        logger.info("🔄 首次初始化 embedding 模型（单例模式）...")
        _global_embedding_model = QwenEmbedding()
        logger.info("✅ Embedding 模型初始化完成，将共享给所有知识库")
    else:
        logger.debug("♻️  复用已存在的 embedding 模型实例")

    # 初始化增强型向量存储（共享同一个embedding模型）
    vector_store = EnhancedQdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=_global_embedding_model
    )

    # 确保集合存在
    vector_store.create_collection_if_not_exists()

    return vector_store


def upsert_md_file(file_path: str, vector_store: EnhancedQdrantVectorStore):
    """上传Markdown文件到Qdrant"""
    source_name = os.path.basename(file_path)
    chunks = parse_markdown_file(file_path)  # 使用原来的函数
    docs = convert_to_langchain_docs(chunks)

    # 准备多向量数据点
    points = []
    for i, doc in enumerate(docs):
        title_vector = vector_store.embedding_model.embed_query(
            f"标题: {doc.metadata['path']}"
        )
        content_vector = vector_store.embedding_model.embed_query(
            f"内容: {doc.metadata['content']}"
        )

        # 生成唯一ID（避免哈希冲突）
        point_id = hash(doc.metadata["id"]) % (2 ** 63)  # 确保ID为正整数

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
    operation_result = vector_store.client.upsert(
        collection_name=vector_store.collection_name,
        points=points,
        wait=False  # 不等待操作完成，避免超时
    )

    # 将 UpdateResult 对象转换为可序列化的字典
    operation_info = {
        "status": str(operation_result.status) if hasattr(operation_result, 'status') else "completed",
        "operation_id": str(operation_result.operation_id) if hasattr(operation_result, 'operation_id') else None,
        "points_count": len(points),
        "collection_name": vector_store.collection_name,
        "source_name": source_name
    }

    print(f"✅ 已上传 {len(points)} 个文档块到集合 {vector_store.collection_name}")
    print(f"�� 操作详情: {operation_info}")
    return operation_info


def delete_by_source(source_name: str, vector_store: EnhancedQdrantVectorStore):
    """按来源名称删除文档"""
    filter_condition = Filter(
        must=[
            FieldCondition(
                key="metadata.source",
                match=MatchValue(value=source_name)
            )
        ]
    )

    # 执行删除
    operation_result = vector_store.client.delete(
        collection_name=vector_store.collection_name,
        points_selector=filter_condition
    )

    # 将 UpdateResult 对象转换为可序列化的字典
    operation_info = {
        "status": str(operation_result.status) if hasattr(operation_result, 'status') else "completed",
        "operation_id": str(operation_result.operation_id) if hasattr(operation_result, 'operation_id') else None,
        "collection_name": vector_store.collection_name,
        "source_name": source_name
    }

    print(f"✅ 已删除来源为 {source_name} 的所有文档块")
    print(f"�� 操作详情: {operation_info}")
    return operation_info


def list_all_collections(host: str = "localhost", port: int = 6333):
    """列出所有集合"""
    # 临时禁用代理，避免对本地连接造成影响
    old_http_proxy = os.environ.pop('http_proxy', None)
    old_https_proxy = os.environ.pop('https_proxy', None)
    old_HTTP_PROXY = os.environ.pop('HTTP_PROXY', None)
    old_HTTPS_PROXY = os.environ.pop('HTTPS_PROXY', None)

    try:
        client = QdrantClient(host=host, port=port)
        collections = client.get_collections().collections
        print(" 当前Qdrant中的集合:")
        for collection in collections:
            print(f"- {collection.name}")
        return collections
    finally:
        # 恢复代理环境变量
        if old_http_proxy:
            os.environ['http_proxy'] = old_http_proxy
        if old_https_proxy:
            os.environ['https_proxy'] = old_https_proxy
        if old_HTTP_PROXY:
            os.environ['HTTP_PROXY'] = old_HTTP_PROXY
        if old_HTTPS_PROXY:
            os.environ['HTTPS_PROXY'] = old_HTTPS_PROXY


def get_collection_info(collection_name: str, host: str = "localhost", port: int = 6333):
    """获取集合详细信息"""
    # 临时禁用代理，避免对本地连接造成影响
    old_http_proxy = os.environ.pop('http_proxy', None)
    old_https_proxy = os.environ.pop('https_proxy', None)
    old_HTTP_PROXY = os.environ.pop('HTTP_PROXY', None)
    old_HTTPS_PROXY = os.environ.pop('HTTPS_PROXY', None)

    try:
        client = QdrantClient(host=host, port=port)
        info = client.get_collection(collection_name)
        print(f" 集合 {collection_name} 详情:")
        print(f"点数: {info.points_count}")
        print(f"状态: {info.status}")
        print(f"配置: {info.config}")
        return info
    except Exception as e:
        print(f"❌ 获取集合信息失败: {str(e)}")
        return None
    finally:
        # 恢复代理环境变量
        if old_http_proxy:
            os.environ['http_proxy'] = old_http_proxy
        if old_https_proxy:
            os.environ['https_proxy'] = old_https_proxy
        if old_HTTP_PROXY:
            os.environ['HTTP_PROXY'] = old_HTTP_PROXY
        if old_HTTPS_PROXY:
            os.environ['HTTPS_PROXY'] = old_HTTPS_PROXY
def upsert_md_file_with_source(file_path: str, vector_store: EnhancedQdrantVectorStore, source_name: str):
    """上传Markdown文件到Qdrant，使用指定的source名称"""
    chunks = parse_markdown_file(file_path)  # 使用原来的函数
    docs = convert_to_langchain_docs(chunks)
    
    # 修改所有文档的source字段为指定的source_name
    for doc in docs:
        doc.metadata['source'] = source_name

    # 准备多向量数据点
    points = []
    for i, doc in enumerate(docs):
        title_vector = vector_store.embedding_model.embed_query(
            f"标题: {doc.metadata['path']}"
        )
        content_vector = vector_store.embedding_model.embed_query(
            f"内容: {doc.metadata['content']}"
        )

        # 生成唯一ID（避免哈希冲突）
        point_id = hash(doc.metadata["id"]) % (2 ** 63)  # 确保ID为正整数

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
    operation_info = vector_store.client.upsert(
        collection_name=vector_store.collection_name,
        points=points,
        wait=True  # 等待操作完成，确保数据被正确索引
    )

    print(f"✅ 已上传 {len(points)} 个文档块到集合 {vector_store.collection_name}")
    print(f" 操作详情: {operation_info}")
    return operation_info


def upsert_md_file_with_original(file_path: str, vector_store: EnhancedQdrantVectorStore, original_file_info: dict = None):
    """上传Markdown文件到Qdrant，同时存储原文件内容用于预览"""
    source_name = os.path.basename(file_path)
    chunks = parse_markdown_file(file_path)  # 使用原来的函数
    docs = convert_to_langchain_docs(chunks)
    
    # 读取原文件完整内容
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        print(f"⚠️ 读取原文件内容失败: {e}")
        original_content = ""

    # 准备多向量数据点（逐个处理，避免显存不足）
    points = []
    total_docs = len(docs)
    
    logger.info(f"🔄 开始处理 {total_docs} 个文档块（逐个向量化，避免显存不足）")
    
    for i, doc in enumerate(docs):
        logger.info(f"🔄 处理文档块 {i+1}/{total_docs}")
        
        # 每个文档块处理前清理GPU缓存，释放显存
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            if (i + 1) % 10 == 0:  # 每10个文档块输出一次显存信息
                logger.debug(f"🧹 GPU缓存已清理（文档块 {i+1}），当前显存: 已分配={allocated:.2f}GiB, 已缓存={reserved:.2f}GiB")
        
        # 单个文档块向量化标题和内容
        title_text = f"标题: {doc.metadata['path']}"
        content_text = f"内容: {doc.metadata['content']}"
        
        # 逐个生成向量
        try:
            title_vector = vector_store.embedding_model.embed_documents([title_text])[0]
            content_vector = vector_store.embedding_model.embed_documents([content_text])[0]
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "CUDA" in error_msg:
                logger.error(f"❌ 文档块 {i+1} 向量化时显存不足: {error_msg}")
                # 清理GPU缓存后重试一次
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    import time
                    time.sleep(2)  # 等待GPU释放显存
                try:
                    title_vector = vector_store.embedding_model.embed_documents([title_text])[0]
                    content_vector = vector_store.embedding_model.embed_documents([content_text])[0]
                    logger.info(f"✅ 文档块 {i+1} 重试成功")
                except Exception as retry_error:
                    logger.error(f"❌ 文档块 {i+1} 重试也失败，跳过该文档块: {retry_error}")
                    continue  # 跳过当前文档块，继续处理下一个
            else:
                # 其他错误，直接抛出
                raise
        
        # 生成唯一ID（避免哈希冲突）
        point_id = hash(doc.metadata["id"]) % (2 ** 63)  # 确保ID为正整数

        # 合并原文件信息到 metadata
        metadata = doc.metadata.copy()
        if original_file_info:
            metadata.update(original_file_info)

        points.append(PointStruct(
            id=point_id,
            vector={
                "title": title_vector,
                "content": content_vector
            },
            payload={
                "page_content": doc.page_content,
                "metadata": metadata,
                "original_content": original_content,  # 添加原文件内容
                "source_name": source_name  # 添加源文件名
            }
        ))
        
        # 每10个文档块处理后清理GPU缓存，释放显存
        if (i + 1) % 10 == 0 and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.debug(f"🧹 已处理 {i+1} 个文档块，已清理GPU缓存")

    # 批量上传
    operation_result = vector_store.client.upsert(
        collection_name=vector_store.collection_name,
        points=points,
        wait=True  # 等待操作完成，确保数据被正确索引
    )

    # 将 UpdateResult 对象转换为可序列化的字典
    operation_info = {
        "status": str(operation_result.status) if hasattr(operation_result, 'status') else "completed",
        "operation_id": str(operation_result.operation_id) if hasattr(operation_result, 'operation_id') else None,
        "points_count": len(points),
        "collection_name": vector_store.collection_name,
        "source_name": source_name
    }

    print(f"✅ 已上传 {len(points)} 个文档块到集合 {vector_store.collection_name}")
    print(f"📄 原文件内容已存储，文件名: {source_name}")
    print(f" 操作详情: {operation_info}")
    
    # 文档索引后清理GPU缓存，释放激活值占用的显存
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.debug(f"🧹 文档索引后已清理GPU缓存")
    
    return operation_info
def upsert_qa_pair(qa_content: str, metadata: Dict[str, Any], vector_store: EnhancedQdrantVectorStore):
    """上传问答对到Qdrant"""
    try:
        # 从metadata中获取文档名，如果没有则生成默认名
        document_name = metadata.get('document_name', f"qa_{hash(qa_content) % 10000}")
        # 确保文档名以.md结尾
        if not document_name.endswith('.md'):
            document_name = f"{document_name}.md"
        
        # 生成唯一ID（基于问题和答案的组合）
        qa_id = hash(f"{metadata.get('question', '')}{metadata.get('answer', '')}") % (2 ** 63)
        
        # 更新metadata中的source字段为文档名
        metadata['source'] = document_name
        
        # 生成向量
        title_vector = vector_store.embedding_model.embed_query(
            f"问题: {metadata.get('question', '')}"
        )
        content_vector = vector_store.embedding_model.embed_query(
            f"内容: {qa_content}"
        )
        
        # 构建数据点
        point = PointStruct(
            id=qa_id,
            vector={
                "title": title_vector,
                "content": content_vector
            },
            payload={
                "page_content": qa_content,
                "metadata": metadata
            }
        )
        
        # 上传到向量数据库
        operation_info = vector_store.client.upsert(
            collection_name=vector_store.collection_name,
            points=[point],
            wait=True
        )
        
        print(f"✅ 已上传问答对到集合 {vector_store.collection_name}")
        print(f"文档名: {document_name}")
        print(f"问题: {metadata.get('question', '')[:50]}...")
        print(f"操作详情: {operation_info}")
        return operation_info
        
    except Exception as e:
        print(f"❌ 上传问答对失败: {str(e)}")
        raise e
