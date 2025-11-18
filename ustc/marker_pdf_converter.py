# marker_pdf_converter.py
import os
import logging
import signal
import threading
from typing import Dict, Any, Optional
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# 配置日志
logger = logging.getLogger("marker_pdf_converter")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def convert_pdf_to_markdown_with_marker(
    file_path: str, 
    output_dir: Optional[str] = None,
    use_llm: bool = False,
    force_ocr: bool = False,
    output_format: str = "markdown",
    base_name: Optional[str] = None,
    timeout: int = 300  # 5分钟超时
) -> Dict[str, Any]:
    """
    使用marker将多种文档格式转换为Markdown
    
    支持格式: PDF, Word, PowerPoint, Excel, 图片等
    
    参数:
    - file_path: 文档文件路径
    - output_dir: 输出目录，如果为None则使用文件所在目录
    - use_llm: 是否使用LLM提高准确性
    - force_ocr: 是否强制OCR处理
    - output_format: 输出格式 ("markdown", "json", "html", "chunks")
    - base_name: 输出文件的基本名称
    - timeout: 超时时间（秒）
    
    返回:
    - 包含转换结果的字典
    """
    try:
        logger.info(f"🔄 开始使用marker转换文档: {file_path}")
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(file_path))
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建marker转换器
        logger.info("🔄 正在初始化marker转换器...")
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        
        # 执行转换（带超时处理）
        logger.info("🔄 正在转换文档...")
        
        # 使用线程和超时机制，添加信号处理
        result_container = {}
        exception_container = {}
        
        def convert_worker():
            try:
                result_container['rendered'] = converter(file_path)
            except BrokenPipeError as e:
                logger.warning("⚠️ 检测到管道中断，尝试重新转换...")
                # 重试一次
                try:
                    result_container['rendered'] = converter(file_path)
                except Exception as retry_error:
                    exception_container['error'] = retry_error
            except Exception as e:
                exception_container['error'] = e
        
        # 启动转换线程
        convert_thread = threading.Thread(target=convert_worker)
        convert_thread.daemon = True
        convert_thread.start()
        
        # 等待转换完成或超时
        convert_thread.join(timeout=timeout)
        
        if convert_thread.is_alive():
            logger.error(f"❌ 文档转换超时（{timeout}秒）")
            return {
                "success": False,
                "message": f"文档转换超时（{timeout}秒），请尝试处理更小的文件",
                "data": {
                    "file_path": file_path,
                    "timeout": timeout
                }
            }
        
        if 'error' in exception_container:
            raise exception_container['error']
        
        if 'rendered' not in result_container:
            raise Exception("转换过程中未返回结果")
        
        rendered = result_container['rendered']
        
        # 提取文本和图像
        text, metadata, images = text_from_rendered(rendered)
        
        # 生成输出文件名
        if base_name is None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 保存markdown文件
        if output_format == "markdown":
            md_file_path = os.path.join(output_dir, f"{base_name}.md")
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"✅ Markdown文件已保存: {md_file_path}")
            
            # 保存图像
            if images:
                images_dir = os.path.join(output_dir, f"{base_name}_images")
                os.makedirs(images_dir, exist_ok=True)
                for img_name, img_data in images.items():
                    img_path = os.path.join(images_dir, img_name)
                    # 处理PIL Image对象
                    if hasattr(img_data, 'save'):
                        img_data.save(img_path)
                    else:
                        with open(img_path, 'wb') as f:
                            f.write(img_data)
                logger.info(f"✅ 图像已保存到: {images_dir}")
        
        # 保存JSON格式（如果需要）
        elif output_format == "json":
            json_file_path = os.path.join(output_dir, f"{base_name}.json")
            import json
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(rendered.model_dump(), f, ensure_ascii=False, indent=2)
            logger.info(f"✅ JSON文件已保存: {json_file_path}")
        
        # 保存HTML格式（如果需要）
        elif output_format == "html":
            html_file_path = os.path.join(output_dir, f"{base_name}.html")
            with open(html_file_path, 'w', encoding='utf-8') as f:
                f.write(rendered.html)
            logger.info(f"✅ HTML文件已保存: {html_file_path}")
        
        return {
            "success": True,
            "message": f"文档转换成功: {file_path}",
            "data": {
                "file_path": file_path,
                "output_dir": output_dir,
                "text_length": len(text),
                "images_count": len(images) if images else 0,
                "metadata": metadata,
                "output_format": output_format
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Marker文档转换失败: {str(e)}")
        
        # 尝试使用备用方法
        logger.info("🔄 尝试使用备用处理方法...")
        try:
            return convert_pdf_with_fallback(file_path, output_dir, base_name)
        except Exception as fallback_error:
            logger.error(f"❌ 备用转换也失败: {str(fallback_error)}")
            return {
                "success": False,
                "message": f"文档转换失败: {str(e)}。备用方法也失败: {str(fallback_error)}",
                "data": {
                    "file_path": file_path,
                    "error": str(e),
                    "fallback_error": str(fallback_error)
                }
            }


def convert_pdf_with_fallback(pdf_path: str, output_dir: Optional[str] = None, base_name: Optional[str] = None) -> Dict[str, Any]:
    """
    备用的PDF处理方法，使用PyPDF2或pdfplumber
    """
    try:
        logger.info("🔄 使用备用方法处理PDF...")
        
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(pdf_path))
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        if base_name is None:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # 尝试使用pdfplumber（优化版本）
        try:
            import pdfplumber
            
            text_content = []
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"📄 PDF共有 {total_pages} 页，开始提取文本...")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"🔄 正在处理第 {page_num}/{total_pages} 页...")
                    
                    # 提取文本
                    page_text = page.extract_text()
                    if page_text:
                        # 清理文本
                        page_text = page_text.strip()
                        if page_text:
                            text_content.append(f"## 第 {page_num} 页\n\n{page_text}\n")
                    
                    # 提取表格（如果有）
                    tables = page.extract_tables()
                    if tables:
                        for table_num, table in enumerate(tables, 1):
                            if table and len(table) > 1:  # 确保表格有内容
                                table_md = f"\n### 表格 {table_num}\n\n"
                                # 转换表格为Markdown格式
                                if len(table) > 0:
                                    # 表头
                                    header = table[0]
                                    if header:
                                        table_md += "| " + " | ".join(str(cell or "") for cell in header) + " |\n"
                                        table_md += "| " + " | ".join("---" for _ in header) + " |\n"
                                    
                                    # 表格内容
                                    for row in table[1:]:
                                        if row:
                                            table_md += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"
                                
                                text_content.append(table_md + "\n")
            
            markdown_content = "\n".join(text_content)
            logger.info(f"✅ 文本提取完成，共提取 {len(text_content)} 个内容块")
            
        except ImportError:
            # 如果pdfplumber不可用，使用PyPDF2
            try:
                import PyPDF2
                
                text_content = []
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(f"## 第 {page_num} 页\n\n{page_text}\n")
                
                markdown_content = "\n".join(text_content)
                
            except ImportError:
                raise Exception("既没有安装pdfplumber也没有安装PyPDF2，无法处理PDF文件")
        
        # 保存markdown文件
        md_file_path = os.path.join(output_dir, f"{base_name}.md")
        with open(md_file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"✅ 备用方法转换成功: {md_file_path}")
        
        return {
            "success": True,
            "message": f"PDF转换成功（备用方法）: {pdf_path}",
            "data": {
                "pdf_path": pdf_path,
                "output_dir": output_dir,
                "text_length": len(markdown_content),
                "images_count": 0,
                "method": "fallback",
                "output_format": "markdown"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ 备用PDF转换失败: {str(e)}")
        raise e


def convert_pdf_to_markdown_simple(pdf_path: str, output_dir: Optional[str] = None) -> str:
    """
    简单的PDF转Markdown函数，返回markdown文本
    
    参数:
    - pdf_path: PDF文件路径
    - output_dir: 输出目录
    
    返回:
    - markdown文本内容
    """
    result = convert_pdf_to_markdown_with_marker(pdf_path, output_dir)
    
    if result["success"]:
        # 读取生成的markdown文件
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        if output_dir is None:
            output_dir = os.path.dirname(pdf_path)
        md_file_path = os.path.join(output_dir, f"{base_name}.md")
        
        if os.path.exists(md_file_path):
            with open(md_file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    return ""


if __name__ == "__main__":
    # 测试代码
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python marker_pdf_converter.py <PDF文件路径> [输出目录]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = convert_pdf_to_markdown_with_marker(pdf_path, output_dir)
    print(f"转换结果: {result}")
