#!/usr/bin/env python3
"""
重新索引所有文档的脚本
清空现有集合并使用新模型（1024维）重新索引所有文档
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import glob
from chunks2embedding import embedding_init
from web_memory import process_pdf_file, process_document_with_marker, process_markdown_file, process_text_file
from fastapi import UploadFile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_and_recreate_collection(collection_name: str, vector_size: int = 1024):
    """清空并重新创建集合"""
    client = QdrantClient(host='localhost', port=6333, check_compatibility=False)
    
    try:
        # 检查集合是否存在
        try:
            info = client.get_collection(collection_name)
            logger.info(f"集合 '{collection_name}' 存在，点数: {info.points_count}")
            
            # 删除现有集合
            logger.info(f"正在删除集合 '{collection_name}'...")
            client.delete_collection(collection_name)
            logger.info(f"✅ 集合 '{collection_name}' 已删除")
        except Exception as e:
            logger.info(f"集合 '{collection_name}' 不存在或已删除: {e}")
        
        # 创建新集合（1024维）
        logger.info(f"正在创建新集合 '{collection_name}' (向量维度: {vector_size})...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "title": VectorParams(size=vector_size, distance=Distance.COSINE),
                "content": VectorParams(size=vector_size, distance=Distance.COSINE)
            }
        )
        logger.info(f"✅ 新集合 '{collection_name}' 创建成功")
        
    except Exception as e:
        logger.error(f"❌ 清空并重新创建集合失败: {e}")
        raise

def find_all_documents(knowledge_base: str):
    """查找知识库中的所有文档"""
    documents = []
    
    # 1. 查找原始文件目录
    original_dir = f"/home/user/ustcchat/ustc/original_files/{knowledge_base}"
    if os.path.exists(original_dir):
        for file_path in glob.glob(os.path.join(original_dir, "*")):
            if os.path.isfile(file_path):
                documents.append({
                    "type": "original",
                    "path": file_path,
                    "filename": os.path.basename(file_path)
                })
    
    # 2. 查找marker输出目录
    marker_dir = "/home/user/ustcchat/ustc/marker_outputs"
    if os.path.exists(marker_dir):
        for doc_dir in glob.glob(os.path.join(marker_dir, "*")):
            if os.path.isdir(doc_dir):
                md_file = os.path.join(doc_dir, f"{os.path.basename(doc_dir)}.md")
                if os.path.exists(md_file):
                    documents.append({
                        "type": "marker",
                        "path": md_file,
                        "filename": os.path.basename(doc_dir) + ".md"
                    })
    
    # 3. 查找DeepSeek OCR输出目录
    deepseek_dir = "/home/user/deepseekocr/output"
    if os.path.exists(deepseek_dir):
        for md_file in glob.glob(os.path.join(deepseek_dir, "*.md")):
            documents.append({
                "type": "deepseek",
                "path": md_file,
                "filename": os.path.basename(md_file)
            })
    
    return documents

def reindex_document(doc_info: dict, knowledge_base: str, vector_store=None):
    """重新索引单个文档"""
    file_path = doc_info["path"]
    filename = doc_info["filename"]
    doc_type = doc_info["type"]
    
    logger.info(f"\\n处理文档: {filename} (类型: {doc_type})")
    
    try:
        import torch
        
        # 根据文件类型处理
        file_extension = os.path.splitext(filename)[1].lower()
        
        # 对于marker和deepseek类型的文档，直接处理markdown文件
        if doc_type in ["marker", "deepseek"]:
            if file_extension in ['.md', '.markdown']:
                # 如果提供了vector_store，直接使用，避免重复加载模型
                if vector_store:
                    from md2chunks import parse_markdown_file_api
                    from chunks2embedding import upsert_md_file_with_original
                    chunks = parse_markdown_file_api(file_path)
                    if chunks:
                        original_file_info = {
                            "original_filename": filename,
                            "original_file_path": None,
                            "file_type": file_extension
                        }
                        operation_info = upsert_md_file_with_original(file_path, vector_store, original_file_info=original_file_info)
                        # 清理GPU缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.info(f"✅ 文档 '{filename}' 索引成功")
                        return True
                    else:
                        logger.warning(f"⚠️ 文档 '{filename}' 没有提取到chunks")
                        return False
                else:
                    result = process_markdown_file(file_path, knowledge_base, filename)
            else:
                logger.warning(f"⚠️ 文档类型为{doc_type}但扩展名不是.md: {file_extension}")
                return False
        else:
            # 对于原始文件，需要根据扩展名处理
            with open(file_path, "rb") as f:
                file_obj = UploadFile(
                    filename=filename,
                    file=f,
                    headers={"content-type": "application/octet-stream"}
                )
                
                if file_extension == '.pdf':
                    result = process_pdf_file(file_path, knowledge_base, filename)
                elif file_extension in ['.md', '.markdown']:
                    result = process_markdown_file(file_path, knowledge_base, filename)
                elif file_extension == '.txt':
                    result = process_text_file(file_path, knowledge_base, filename)
                elif file_extension in ['.docx', '.ppt', '.pptx', '.xls', '.xlsx']:
                    result = process_document_with_marker(file_path, knowledge_base, filename)
                else:
                    logger.warning(f"⚠️ 不支持的文件类型: {file_extension}")
                    return False
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if result.get("success"):
            logger.info(f"✅ 文档 '{filename}' 索引成功")
            return True
        else:
            logger.error(f"❌ 文档 '{filename}' 索引失败: {result.get('message', '未知错误')}")
            return False
                
    except Exception as e:
        logger.error(f"❌ 处理文档 '{filename}' 时出错: {e}", exc_info=True)
        # 即使出错也清理GPU缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        return False

def main():
    """主函数"""
    knowledge_base = "NSRL文档库"
    
    logger.info("=" * 100)
    logger.info("开始重新索引所有文档")
    logger.info("=" * 100)
    
    # 1. 清空并重新创建集合
    logger.info(f"\\n步骤1: 清空并重新创建集合 '{knowledge_base}'")
    clear_and_recreate_collection(knowledge_base, vector_size=1024)
    
    # 2. 查找所有文档
    logger.info(f"\\n步骤2: 查找知识库 '{knowledge_base}' 中的所有文档")
    documents = find_all_documents(knowledge_base)
    logger.info(f"找到 {len(documents)} 个文档")
    
    if not documents:
        logger.warning("⚠️ 没有找到任何文档，请检查文档路径")
        return
    
    # 3. 重新索引所有文档
    logger.info(f"\\n步骤3: 开始重新索引 {len(documents)} 个文档")
    
    # 预先初始化vector_store，避免每个文档都重新加载模型
    logger.info("\\n预先初始化embedding模型（只加载一次）...")
    vector_store = embedding_init(collection_name=knowledge_base)
    logger.info("✅ Embedding模型已加载")
    
    success_count = 0
    failed_count = 0
    
    import torch
    
    for i, doc_info in enumerate(documents, 1):
        logger.info(f"\\n[{i}/{len(documents)}] 处理文档: {doc_info['filename']}")
        
        # 检查GPU显存使用情况
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            if allocated > 20:  # 如果已分配超过20GB，清理缓存
                logger.warning(f"⚠️ GPU显存使用较高 ({allocated:.2f}GB)，清理缓存...")
                torch.cuda.empty_cache()
        
        if reindex_document(doc_info, knowledge_base, vector_store=vector_store):
            success_count += 1
        else:
            failed_count += 1
        
        # 每处理5个文档清理一次GPU缓存（更频繁）
        if i % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"已处理 {i} 个文档，清理GPU缓存 (当前显存: {allocated:.2f}GB)")
    
    # 4. 输出结果
    logger.info("\\n" + "=" * 100)
    logger.info("重新索引完成")
    logger.info("=" * 100)
    logger.info(f"成功: {success_count} 个文档")
    logger.info(f"失败: {failed_count} 个文档")
    logger.info(f"总计: {len(documents)} 个文档")
    
    # 5. 验证结果
    from qdrant_client import QdrantClient
    client = QdrantClient(host='localhost', port=6333, check_compatibility=False)
    info = client.get_collection(knowledge_base)
    logger.info(f"\\n集合 '{knowledge_base}' 最终状态:")
    logger.info(f"  点数: {info.points_count}")
    logger.info(f"  向量维度: {info.config.params.vectors}")

if __name__ == "__main__":
    main()

