#!/usr/bin/env python3
"""
使用ModelScope下载marker所需的模型
"""

import os
import sys
from modelscope import snapshot_download
from pathlib import Path

def download_marker_models():
    """下载marker所需的模型"""
    
    # 设置ModelScope缓存目录
    cache_dir = Path.home() / ".cache" / "datalab" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # marker需要的模型列表
    models = {
        "layout": "AI-ModelScope/surya-layout",
        "text_recognition": "AI-ModelScope/surya-ocr",
        "table_structure": "AI-ModelScope/surya-table-structure",
        "table_cell": "AI-ModelScope/surya-table-cell"
    }
    
    print("🚀 开始使用ModelScope下载marker模型...")
    
    for model_type, model_id in models.items():
        print(f"\n📥 正在下载 {model_type} 模型: {model_id}")
        
        try:
            # 设置模型保存路径
            model_dir = cache_dir / model_type
            
            # 下载模型
            downloaded_path = snapshot_download(
                model_id=model_id,
                cache_dir=str(cache_dir),
                local_dir=str(model_dir)
            )
            
            print(f"✅ {model_type} 模型下载完成: {downloaded_path}")
            
        except Exception as e:
            print(f"❌ {model_type} 模型下载失败: {str(e)}")
            continue
    
    print("\n🎉 模型下载完成！")
    
    # 验证模型是否下载成功
    print("\n📋 验证下载的模型:")
    for model_type in models.keys():
        model_dir = cache_dir / model_type
        if model_dir.exists():
            files = list(model_dir.rglob("*"))
            print(f"  {model_type}: {len(files)} 个文件")
        else:
            print(f"  {model_type}: 未找到")

if __name__ == "__main__":
    download_marker_models()
