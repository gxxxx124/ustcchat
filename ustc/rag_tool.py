from langchain.tools import Tool
from chunks2embedding import embedding_init
from pydantic import BaseModel
import json
import logging
import os

logger = logging.getLogger(__name__)

# 原文件存储目录
ORIGINAL_FILES_DIR = "/home/user/ustcchat/original_files"


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
    
    # 保存collection_name供后续使用（闭包变量）
    _collection_name = collection_name

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
        # 存储引用信息，用于后续生成引用链接
        references = []
        
        # 检查是否需要返回完整文档
        # 如果最高相似度结果相似度高且文档短，尝试获取完整文档
        use_full_document = False
        full_document_content = None
        full_document_source = None
        
        if results:
            top_result = results[0]
            top_score = top_result[1]
            top_doc = top_result[0]
            top_metadata = top_doc.metadata
            
            # 条件：相似度 >= 0.7 且不是QA对
            if (top_score >= 0.7 and 
                not top_metadata.get('is_qa_pair') and 
                top_metadata.get('type') != 'qa'):
                
                # 尝试从本地文件系统加载完整文档内容
                try:
                    source = top_metadata.get('source', '')
                    if source:
                        # 获取原始文件路径（从metadata中）
                        original_file_path = top_metadata.get('original_file_path', '')
                        
                        # 如果没有original_file_path，尝试构建路径
                        if not original_file_path:
                            # 从source构建：source通常是"文档名.md"，需要找到对应的原始文件
                            # 先尝试从知识库名称和文档名构建路径
                            document_name = source
                            if document_name.endswith('.md'):
                                # 去掉.md后缀，尝试找到原始文件
                                base_name = document_name[:-3]
                                # 尝试常见的原始文件扩展名
                                possible_extensions = ['.pdf', '.docx', '.doc', '.txt', '.md']
                                
                                # 构建可能的文件路径
                                kb_dir = os.path.join(ORIGINAL_FILES_DIR, _collection_name)
                                if os.path.exists(kb_dir):
                                    for ext in possible_extensions:
                                        possible_path = os.path.join(kb_dir, base_name + ext)
                                        if os.path.exists(possible_path):
                                            original_file_path = possible_path
                                            break
                                    
                                    # 如果还是没找到，尝试直接匹配文件名（包含扩展名）
                                    if not original_file_path:
                                        for filename in os.listdir(kb_dir):
                                            if filename.startswith(base_name) or base_name in filename:
                                                original_file_path = os.path.join(kb_dir, filename)
                                                break
                            else:
                                # source不是.md格式，直接尝试作为文件名
                                kb_dir = os.path.join(ORIGINAL_FILES_DIR, _collection_name)
                                possible_path = os.path.join(kb_dir, source)
                                if os.path.exists(possible_path):
                                    original_file_path = possible_path
                        else:
                            # 如果有original_file_path，转换为绝对路径
                            if original_file_path.startswith("original_files/"):
                                original_file_path = os.path.join("/home/user/ustcchat", original_file_path)
                        
                        # 尝试读取本地文件
                        if original_file_path and os.path.exists(original_file_path):
                            # 根据文件类型读取内容
                            file_ext = os.path.splitext(original_file_path)[1].lower()
                            
                            if file_ext == '.md' or file_ext == '.txt':
                                # 文本文件，直接读取
                                with open(original_file_path, 'r', encoding='utf-8') as f:
                                    full_document_content = f.read()
                            elif file_ext == '.pdf':
                                # PDF文件，尝试读取markdown版本
                                # 1. 先尝试在deepseek输出目录查找
                                from pathlib import Path
                                pdf_name = Path(original_file_path).stem
                                deepseek_output_dir = Path.home() / "deepseekocr" / "output"
                                mmd_file = deepseek_output_dir / f"{pdf_name}.mmd"
                                
                                if mmd_file.exists():
                                    with open(mmd_file, 'r', encoding='utf-8') as f:
                                        full_document_content = f.read()
                                    logger.info(f"✅ 从DeepSeek输出目录读取markdown: {mmd_file}")
                                else:
                                    # 2. 尝试在marker输出目录查找
                                    marker_output_dir = Path("/home/user/ustcchat/ustc/marker_outputs") / pdf_name
                                    md_file = marker_output_dir / f"{pdf_name}.md"
                                    
                                    if md_file.exists():
                                        with open(md_file, 'r', encoding='utf-8') as f:
                                            full_document_content = f.read()
                                        logger.info(f"✅ 从marker输出目录读取markdown: {md_file}")
                                    else:
                                        # PDF没有对应的md文件，跳过
                                        logger.info(f"⚠️ PDF文件 {original_file_path} 没有对应的markdown文件，跳过完整文档返回")
                                        full_document_content = None
                            else:
                                # 其他格式，尝试读取文本（如果可能）
                                try:
                                    with open(original_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                        full_document_content = f.read()
                                except:
                                    logger.warning(f"⚠️ 无法读取文件 {original_file_path}，跳过完整文档返回")
                                    full_document_content = None
                            
                            if full_document_content:
                                full_document_source = source
                                
                                # 检查文档长度（字符数）
                                doc_length = len(full_document_content)
                                # 如果文档长度 < 10000字符，使用完整文档
                                if doc_length < 10000:
                                    use_full_document = True
                                    logger.info(f"✅ 检测到短文档（{doc_length}字符），将从本地文件返回完整文档内容: {original_file_path}")
                                else:
                                    logger.info(f"⚠️ 文档长度 {doc_length} 字符超过10000字符阈值，使用chunk模式")
                        else:
                            logger.info(f"⚠️ 未找到本地文件: {original_file_path if original_file_path else '路径未构建'}")
                except Exception as e:
                    logger.warning(f"⚠️ 从本地加载完整文档失败: {str(e)}")
                    use_full_document = False
        
        # 如果使用完整文档，先添加完整文档，然后继续添加其他结果
        if use_full_document and full_document_content:
            document_name = full_document_source
            if document_name.endswith('.md'):
                document_name = document_name[:-3]
            
            formatted_results.append(
                f"【完整文档 - 相似度: {top_score:.4f}】\n"
                f"来源: {full_document_source}\n"
                f"文档名: {document_name}\n"
                f"完整内容:\n{full_document_content}\n"
                "----------------------------------------"
            )
            
            # 记录引用信息
            ref_info = {
                "document_name": document_name,
                "source": full_document_source,
                "title": document_name,
                "page_info": "完整文档",
                "score": top_score
            }
            references.append(ref_info)
            regular_docs_found = 1
            
            # 继续处理其他结果（跳过第一个，因为已经作为完整文档返回）
            result_counter = 1  # 从1开始，因为完整文档是结果#1
            for doc, score in results[1:]:  # 跳过第一个结果
                metadata = doc.metadata
                source = metadata.get('source', '未知')
                title = metadata.get('title', metadata.get('title_text', '无标题'))
                
                # 如果这个结果来自同一个文档，跳过（避免重复）
                if source == full_document_source:
                    continue
                
                # 提取文档名（去掉.md后缀）
                document_name = source
                if document_name.endswith('.md'):
                    document_name = document_name[:-3]
                
                # 提取页码或章节信息
                page_info = ""
                if 'page' in metadata:
                    page_info = f"第{metadata['page']}页"
                elif 'level' in metadata:
                    page_info = f"章节: {title}"
                
                # 记录引用信息
                ref_info = {
                    "document_name": document_name,
                    "source": source,
                    "title": title,
                    "page_info": page_info,
                    "score": score
                }
                references.append(ref_info)
                
                # 检查是否为QA对
                if metadata.get('is_qa_pair') or metadata.get('type') == 'qa':
                    qa_pairs_found += 1
                    result_counter += 1
                    formatted_results.append(
                        f"【QA对知识库 - 结果 #{result_counter} (相似度: {score:.4f})】\n"
                        f"来源: {source}\n"
                        f"内容: {doc.page_content}\n"
                        "----------------------------------------"
                    )
                else:
                    regular_docs_found += 1
                    result_counter += 1
                    formatted_results.append(
                        f"【文档片段 - 结果 #{result_counter} (相似度: {score:.4f})】\n"
                        f"标题: {title}\n"
                        f"内容: {doc.page_content}\n"
                        f"来源: {source}\n"
                        "----------------------------------------"
                    )
        else:
            # 正常处理：返回chunks
            result_counter = 0  # 用于编号实际添加的结果
            for doc, score in results:
                metadata = doc.metadata
                source = metadata.get('source', '未知')
                title = metadata.get('title', metadata.get('title_text', '无标题'))
                
                # 提取文档名（去掉.md后缀）
                document_name = source
                if document_name.endswith('.md'):
                    document_name = document_name[:-3]
                
                # 提取页码或章节信息
                page_info = ""
                if 'page' in metadata:
                    page_info = f"第{metadata['page']}页"
                elif 'level' in metadata:
                    page_info = f"章节: {title}"
                
                # 记录引用信息
                ref_info = {
                    "document_name": document_name,
                    "source": source,
                    "title": title,
                    "page_info": page_info,
                    "score": score
                }
                references.append(ref_info)
                
                # 检查是否为QA对（在循环内部检查每个结果）
                if metadata.get('is_qa_pair') or metadata.get('type') == 'qa':
                    qa_pairs_found += 1
                    result_counter += 1
                    formatted_results.append(
                        f"【QA对知识库 - 结果 #{result_counter} (相似度: {score:.4f})】\n"
                        f"来源: {source}\n"
                        f"内容: {doc.page_content}\n"
                        "----------------------------------------"
                    )
                else:
                    regular_docs_found += 1
                    result_counter += 1
                    formatted_results.append(
                        f"【文档片段 - 结果 #{result_counter} (相似度: {score:.4f})】\n"
                        f"标题: {title}\n"
                        f"内容: {doc.page_content}\n"
                        f"来源: {source}\n"
                        "----------------------------------------"
                    )
        
        # 添加统计信息
        if use_full_document:
            formatted_results.insert(0, f"📚 检测到短文档且相似度高，返回完整文档内容：\n")
        elif qa_pairs_found > 0:
            formatted_results.insert(0, f"📚 在QA对知识库中找到 {qa_pairs_found} 个相关问答对，{regular_docs_found} 个文档片段：\n")
        else:
            formatted_results.insert(0, f"📚 在知识库中找到 {regular_docs_found} 个相关文档片段：\n")
        
        # 在结果末尾添加引用信息（使用特殊标记，便于后续解析）
        result_text = "\n".join(formatted_results)
        # 添加引用信息标记
        ref_marker = f"\n\n<REFERENCES>{json.dumps(references, ensure_ascii=False)}</REFERENCES>"
        return result_text + ref_marker

    return Tool.from_function(
        name="rag_knowledge_search",
        description="使用RAG系统搜索知识库中的相关信息。仅需要输入查询语句。",
        func=rag_search_tool,
        args_schema=RAGSearchInput,
        return_direct=False
    )