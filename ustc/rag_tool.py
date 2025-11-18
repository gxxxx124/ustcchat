from langchain.tools import Tool
from chunks2embedding import embedding_init
from pydantic import BaseModel


# 简化输入模式
class RAGSearchInput(BaseModel):
    query: str  # 仅接受查询字符串


def create_rag_tool(
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "nsrl_tech_docs"
):
    """
    创建RAG搜索工具的工厂函数

    参数:
    - host: Qdrant服务器主机地址 (例如: "localhost" 或 "http://localhost")
    - port: Qdrant服务器端口 (默认: 6333)
    - collection_name: 集合名称 (默认: "5.chunks")
    """
    # 确保host包含协议
    if not host.startswith(('http://', 'https://')):
        host = f"http://{host}"

    # 初始化向量存储
    vector_store = embedding_init(
        host=host,
        port=port,
        collection_name=collection_name
    )

    def rag_search_tool(query: str) -> str:
        """搜索工具，仅接受查询字符串"""
        # 固定参数 - 增加返回结果数量，提高内容权重
        k = 15  # 从5增加到15，获取更多相关结果
        title_weight = 0.6  # 降低标题权重
        content_weight = 0.4  # 提高内容权重，获取更多相关内容

        results = vector_store.weighted_hybrid_search(
            query=query,
            k=k,
            title_weight=title_weight,
            content_weight=content_weight
        )

        if not results:
            return "未在知识库中找到相关信息。请尝试使用网络搜索获取最新信息。"

        formatted_results = []
        qa_pairs_found = 0
        regular_docs_found = 0
        
        for i, (doc, score) in enumerate(results, 1):
            metadata = doc.metadata
            
            # 检查是否为QA对
            if metadata.get('is_qa_pair') or metadata.get('type') == 'qa':
                qa_pairs_found += 1
                formatted_results.append(
                    f"【QA对知识库 - 结果 #{i} (相似度: {score:.4f})】\n"
                    f"来源: {metadata.get('source', '未知')}\n"
                    f"内容: {doc.page_content}\n"
                    "----------------------------------------"
                )
            else:
                regular_docs_found += 1
                formatted_results.append(
                    f"【文档片段 - 结果 #{i} (相似度: {score:.4f})】\n"
                    f"标题: {metadata.get('title', metadata.get('title_text', '无标题'))}\n"
                    f"内容: {doc.page_content}\n"
                    f"来源: {metadata.get('source', '未知')}\n"
                    "----------------------------------------"
                )
        
        # 添加统计信息
        if qa_pairs_found > 0:
            formatted_results.insert(0, f"📚 在QA对知识库中找到 {qa_pairs_found} 个相关问答对，{regular_docs_found} 个文档片段：\n")
        else:
            formatted_results.insert(0, f"📚 在知识库中找到 {regular_docs_found} 个相关文档片段：\n")
        
        return "\n".join(formatted_results)

    return Tool.from_function(
        name="rag_knowledge_search",
        description="使用RAG系统搜索知识库中的相关信息。仅需要输入查询语句。",
        func=rag_search_tool,
        args_schema=RAGSearchInput,
        return_direct=False
    )