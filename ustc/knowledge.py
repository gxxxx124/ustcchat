from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import logging
import re
import uuid  # 用于生成唯一task_id
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchText
from embedding import QwenEmbedding
from chunks2embedding import embedding_init

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("knowledge_search")

# 创建FastAPI应用
app = FastAPI(title="知识库检索系统", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# 数据模型
class SearchRequest(BaseModel):
    query: str
    knowledge_base_name: str
    search_type: str = "hybrid"  # "vector", "keyword", "hybrid"
    top_k: Optional[int] = 20  # 从10增加到20，获取更多结果
    similarity_threshold: Optional[float] = 0.3  # 从0.5降低到0.3，获取更多相关结果
    keyword_match_threshold: Optional[int] = 1
    task_id: Optional[str] = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")  # 自动生成task_id


class SearchResult(BaseModel):
    content: str
    document_name: str
    title: str
    score: float
    search_type: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    success: bool
    message: str
    query: str
    knowledge_base_name: str
    search_type: str
    task_id: str  # 添加task_id到响应
    results: List[SearchResult]
    total_results: int


class KnowledgeSearchSystem:
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.embedding_model = QwenEmbedding()

    def vector_search(self, query: str, kb_name: str, top_k: int = 10, similarity_threshold: float = 0.5):
        try:
            logger.info(f"执行向量搜索 (task: {query})")
            vector_store = embedding_init(collection_name=kb_name)
            search_results = vector_store.weighted_hybrid_search(query=query, k=top_k)
            results = []
            for doc, score in search_results:
                if score < similarity_threshold:
                    continue
                document_name = doc.metadata.get("source", "未知文档")
                if document_name.endswith('.md'):
                    document_name = document_name[:-3]
                title = doc.metadata.get("title", "无标题")
                result = SearchResult(
                    content=doc.page_content,
                    document_name=document_name,
                    title=title,
                    score=score,
                    search_type="vector",
                    metadata=doc.metadata
                )
                results.append(result)
            logger.info(f"向量搜索找到 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            return []

    def keyword_search(self, query: str, kb_name: str, top_k: int = 10, keyword_match_threshold: int = 1):
        try:
            logger.info(f"执行关键词搜索 (task: {query})")
            keywords = self._extract_keywords(query)
            if not keywords:
                logger.warning(f"未提取到有效关键词: {query}")
                return []

            # 获取所有文档块
            all_points = []
            offset = None
            while True:
                points, next_offset = self.client.scroll(
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

            results = []
            logger.info(f"检查 {len(all_points)} 个文档块进行关键词匹配")

            for i, point in enumerate(all_points):
                # 从metadata中获取content
                metadata = point.payload.get("metadata", {})
                content = metadata.get("content", "")

                # 添加调试信息
                if i < 3:
                    logger.debug(f"文档 {i + 1} 内容片段: '{content[:100]}...'")
                    logger.debug(f"是否包含 'NGS': {'NGS' in content}")

                # 清理HTML标签
                clean_content = re.sub(r'<[^>]+>', '', content)
                # 清理特殊字符
                clean_content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', clean_content)

                # 添加调试
                if i < 3:
                    logger.debug(f"清理后内容片段: '{clean_content[:100]}...'")
                    logger.debug(f"清理后是否包含 'NGS': {'NGS' in clean_content}")

                keyword_score = sum(1 for kw in keywords if kw in clean_content)

                if keyword_score >= keyword_match_threshold:
                    # 从metadata中获取source
                    document_name = metadata.get("source", "未知文档")
                    if document_name.endswith('.md'):
                        document_name = document_name[:-3]
                    title = metadata.get("title", "无标题")
                    search_result = SearchResult(
                        content=content,
                        document_name=document_name,
                        title=title,
                        score=keyword_score / len(keywords),
                        search_type="keyword",
                        metadata=metadata
                    )
                    results.append(search_result)

            logger.info(f"关键词搜索匹配到 {len(results)} 个结果")
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"关键词搜索失败: {str(e)}", exc_info=True)
            return []

    def hybrid_search(self, query: str, kb_name: str, top_k: int = 10, similarity_threshold: float = 0.5,
                      keyword_match_threshold: int = 1):
        vector_results = self.vector_search(query, kb_name, top_k, similarity_threshold)
        keyword_results = self.keyword_search(query, kb_name, top_k, keyword_match_threshold)
        combined_results = []
        seen_docs = set()
        for result in vector_results:
            doc_key = f"{result.document_name}_{result.title}"
            if doc_key not in seen_docs:
                combined_results.append(result)
                seen_docs.add(doc_key)
        for result in keyword_results:
            doc_key = f"{result.document_name}_{result.title}"
            if doc_key not in seen_docs:
                combined_results.append(result)
                seen_docs.add(doc_key)
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return combined_results[:top_k]

    def _extract_keywords(self, query: str) -> List[str]:
        # 修改正则表达式，确保能提取像"NGS"这样的短词
        keywords = re.findall(r'[A-Z]+|\b\w+\b', query)

        stop_words = {'的', '是', '在', '有', '和', '与', '或', '但', '而', '如果', '因为', '所以',
                      'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'so'}

        # 移除长度限制，保留所有有效关键词
        keywords = [kw for kw in keywords if kw not in stop_words]

        logger.debug(f"提取的关键词: {keywords}")
        return keywords

    def _calculate_keyword_score(self, content: str, keywords: List[str]) -> int:
        content_lower = content
        score = 0
        for keyword in keywords:
            if keyword in content_lower:
                score += 1
        return score


# 创建全局搜索系统实例
search_system = KnowledgeSearchSystem()


@app.post("/api/search", response_model=SearchResponse)
async def search_knowledge_base(request: SearchRequest):
    try:
        logger.info(f"收到搜索请求, task_id: {request.task_id}, query: {request.query}")

        try:
            search_system.client.get_collection(request.knowledge_base_name)
        except:
            logger.error(f"知识库 '{request.knowledge_base_name}' 不存在 (task_id: {request.task_id})")
            return SearchResponse(
                success=False,
                message=f"知识库 '{request.knowledge_base_name}' 不存在",
                query=request.query,
                knowledge_base_name=request.knowledge_base_name,
                search_type=request.search_type,
                task_id=request.task_id,  # 返回task_id
                results=[],
                total_results=0
            )

        if request.search_type == "vector":
            results = search_system.vector_search(
                query=request.query,
                kb_name=request.knowledge_base_name,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold
            )
        elif request.search_type == "keyword":
            results = search_system.keyword_search(
                query=request.query,
                kb_name=request.knowledge_base_name,
                top_k=request.top_k,
                keyword_match_threshold=request.keyword_match_threshold
            )
        else:
            results = search_system.hybrid_search(
                query=request.query,
                kb_name=request.knowledge_base_name,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold,
                keyword_match_threshold=request.keyword_match_threshold
            )

        logger.info(f"搜索完成, task_id: {request.task_id}, 找到 {len(results)} 个结果")
        return SearchResponse(
            success=True,
            message=f"{request.search_type}搜索完成，找到 {len(results)} 个相关结果",
            query=request.query,
            knowledge_base_name=request.knowledge_base_name,
            search_type=request.search_type,
            task_id=request.task_id,  # 返回task_id
            results=results,
            total_results=len(results)
        )
    except Exception as e:
        logger.error(f"检索失败 (task_id: {request.task_id}): {str(e)}")
        return SearchResponse(
            success=False,
            message=f"检索失败: {str(e)}",
            query=request.query,
            knowledge_base_name=request.knowledge_base_name,
            search_type=request.search_type,
            task_id=request.task_id,  # 确保即使出错也返回task_id
            results=[],
            total_results=0
        )


@app.get("/health")
async def health_check():
    try:
        search_system.client.get_collections()
        return {"success": True, "message": "知识库检索系统运行正常", "port": 5002}
    except Exception as e:
        return {"success": False, "message": f"系统异常: {str(e)}", "port": 5002}


if __name__ == "__main__":
    import uvicorn

    logger.info(" 启动知识库检索系统...")
    uvicorn.run(app, host="0.0.0.0", port=5002, log_level="info")