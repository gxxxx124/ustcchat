#!/usr/bin/env python3
"""
独立的marker处理脚本，避免Broken pipe问题
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_with_marker_standalone(file_path: str, output_dir: str, base_name: str) -> dict:
    """
    使用独立进程运行marker转换，避免Broken pipe问题
    """
    try:
        logger.info(f"🔄 开始独立marker转换: {file_path}")
        
        # 创建临时脚本
        script_content = f'''
import os
import sys
import json
import tempfile
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, "{os.path.dirname(os.path.abspath(__file__))}")

try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    
    # 创建转换器
    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    
    # 执行转换
    result = converter("{file_path}")
    
    # 提取文本
    text = text_from_rendered(result)
    
    # 确保text是字符串
    if not isinstance(text, str):
        if isinstance(text, tuple) and len(text) > 0:
            text = text[0]  # 取第一个元素
        text = str(text)
    
    # 保存结果
    os.makedirs("{output_dir}", exist_ok=True)
    md_file = os.path.join("{output_dir}", "{base_name}.md")
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    # 返回结果
    result_data = {{
        "success": True,
        "text_length": len(text),
        "md_file": md_file,
        "method": "marker_standalone"
    }}
    
    print(json.dumps(result_data))
    
except Exception as e:
    import traceback
    error_data = {{
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc(),
        "method": "marker_standalone"
    }}
    print(json.dumps(error_data))
    sys.exit(1)
'''
        
        # 创建临时脚本文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            # 运行独立进程
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            # 检查是否有JSON输出
            stdout_lines = result.stdout.strip().split('\n')
            json_output = None
            
            # 查找JSON输出（通常在最后一行）
            for line in reversed(stdout_lines):
                if line.strip().startswith('{') and line.strip().endswith('}'):
                    try:
                        json_output = json.loads(line.strip())
                        break
                    except:
                        continue
            
            if json_output:
                if json_output.get("success"):
                    logger.info(f"✅ Marker独立转换成功: {file_path}")
                    return json_output
                else:
                    raise Exception(f"Marker转换失败: {json_output.get('error', '未知错误')}")
            else:
                # 如果没有JSON输出，检查是否生成了文件
                expected_md_file = os.path.join(output_dir, f"{base_name}.md")
                if os.path.exists(expected_md_file):
                    with open(expected_md_file, 'r', encoding='utf-8') as f:
                        text = f.read()
                    return {
                        "success": True,
                        "text_length": len(text),
                        "md_file": expected_md_file,
                        "method": "marker_standalone"
                    }
                else:
                    raise Exception(f"子进程执行失败，无JSON输出且未生成文件。stdout: {result.stdout[:500]}...")
                
        finally:
            # 清理临时脚本
            try:
                os.unlink(script_path)
            except:
                pass
                
    except subprocess.TimeoutExpired:
        logger.error("❌ Marker转换超时")
        return {
            "success": False,
            "error": "转换超时",
            "method": "marker_standalone"
        }
    except Exception as e:
        logger.error(f"❌ Marker独立转换失败: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "method": "marker_standalone"
        }

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python marker_standalone.py <file_path> <output_dir> <base_name>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_dir = sys.argv[2]
    base_name = sys.argv[3]
    
    result = convert_with_marker_standalone(file_path, output_dir, base_name)
    print(json.dumps(result))