#!/usr/bin/env python3
"""
直接上传 Markdown 文件到知识库的脚本
"""
import asyncio
import os
from chunks2embedding import embedding_init, upsert_md_file

async def upload_md_files():
    """上传所有 Markdown 文件到知识库"""
    
    # 知识库名称
    knowledge_base_name = "NSRL技术文档库"
    
    # Markdown 文件路径列表
    md_files = [
        "/home/easyai/OCRFlux/localworkspace/markdowns/NSRL-AC07-TN-2022-001-v1/NSRL-AC07-TN-2022-001-v1.md",
        "/home/easyai/OCRFlux/localworkspace/markdowns/NSRL-AC07-TN-2022-002-v1/NSRL-AC07-TN-2022-002-v1.md",
        "/home/easyai/OCRFlux/localworkspace/markdowns/NSRL-IT01-TN-2022-001-v1/NSRL-IT01-TN-2022-001-v1.md",
        "/home/easyai/OCRFlux/localworkspace/markdowns/NSRL-IT02-TN-2022-001-v1/NSRL-IT02-TN-2022-001-v1.md"
    ]
    
    print(f"🚀 开始上传文件到知识库: {knowledge_base_name}")
    
    # 初始化向量存储
    print("📚 初始化向量存储...")
    vector_store = embedding_init(collection_name=knowledge_base_name)
    print("✅ 向量存储初始化完成")
    
    # 上传每个文件
    success_count = 0
    for i, file_path in enumerate(md_files, 1):
        if not os.path.exists(file_path):
            print(f"❌ 文件不存在: {file_path}")
            continue
            
        print(f"📄 [{i}/{len(md_files)}] 上传文件: {os.path.basename(file_path)}")
        
        try:
            # 直接调用 upsert_md_file 函数
            operation_info = upsert_md_file(file_path, vector_store)
            print(f"✅ 上传成功: {os.path.basename(file_path)}")
            print(f"   操作信息: {operation_info}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ 上传失败: {os.path.basename(file_path)}")
            print(f"   错误信息: {str(e)}")
    
    print(f"\n🎉 上传完成！成功上传 {success_count}/{len(md_files)} 个文件")

if __name__ == "__main__":
    asyncio.run(upload_md_files())


