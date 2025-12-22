"""
DeepSeek OCR PDF 转 Markdown 转换器
使用 DeepSeek-OCR 模型进行 PDF 转 Markdown 转换
"""
import os
import subprocess
import logging
import shutil
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def pdf2md(pdf_path: str) -> str:
    """
    使用 DeepSeek OCR 将 PDF 转换为 Markdown
    
    参数:
        pdf_path: PDF 文件路径
        
    返回:
        markdown 文件路径
    """
    try:
        # 获取 PDF 文件名（不含扩展名）
        pdf_name = Path(pdf_path).stem
        pdf_dir = Path(pdf_path).parent
        
        # 设置 DeepSeek OCR 环境
        deepseekocr_dir = Path.home() / "deepseekocr"
        deepseek_ocr_vllm = deepseekocr_dir / "DeepSeek-OCR" / "DeepSeek-OCR-main" / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm"
        conda_env = "deepseek-ocr"
        
        # 输出目录
        output_dir = deepseekocr_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 临时配置文件路径
        config_file = deepseek_ocr_vllm / "config.py"
        original_config = None
        
        # 备份并修改配置文件
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                original_config = f.read()
            
            # 使用正则表达式替换 INPUT_PATH 和 OUTPUT_PATH
            # 替换 INPUT_PATH，匹配任何值
            new_config = re.sub(
                r"INPUT_PATH\s*=\s*['\"].*?['\"]",
                f"INPUT_PATH = '{pdf_path}'",
                original_config
            )
            # 替换 OUTPUT_PATH，匹配任何值
            new_config = re.sub(
                r"OUTPUT_PATH\s*=\s*['\"].*?['\"]",
                f"OUTPUT_PATH = '{output_dir}'",
                new_config
            )
            
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_config)
        
        # 设置环境变量
        env = os.environ.copy()
        env["PATH"] = f"/home/user/miniconda3/envs/{conda_env}/bin:" + env.get("PATH", "")
        env["CONDA_PREFIX"] = f"/home/user/miniconda3/envs/{conda_env}"
        env["LD_LIBRARY_PATH"] = f"{env['CONDA_PREFIX']}/lib:" + env.get("LD_LIBRARY_PATH", "")
        
        # 运行转换脚本
        script_path = deepseek_ocr_vllm / "run_dpsk_ocr_pdf.py"
        
        logger.info(f"开始使用 DeepSeek OCR 转换 PDF: {pdf_path}")
        logger.info(f"输出目录: {output_dir}")
        
        # 执行转换
        cmd = [
            f"/home/user/miniconda3/envs/{conda_env}/bin/python",
            str(script_path)
        ]
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(deepseek_ocr_vllm),
            bufsize=1
        )
        
        # 实时输出日志
        for line in iter(process.stdout.readline, ''):
            if line:
                logger.info(f"DeepSeek OCR: {line.strip()}")
        
        return_code = process.wait()
        
        # 恢复原始配置
        if original_config and config_file.exists():
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(original_config)
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd, f"DeepSeek OCR 转换失败，返回码: {return_code}")
        
        # 查找生成的 markdown 文件
        # DeepSeek OCR 输出格式: {pdf_name}.mmd
        expected_md_file = output_dir / f"{pdf_name}.mmd"
        
        if not expected_md_file.exists():
            # 尝试查找其他可能的文件名
            md_files = list(output_dir.glob("*.mmd"))
            if md_files:
                expected_md_file = md_files[0]
                logger.warning(f"未找到预期的 {pdf_name}.mmd，使用找到的文件: {expected_md_file}")
            else:
                raise FileNotFoundError(f"DeepSeek OCR 转换完成，但未找到生成的 markdown 文件。输出目录: {output_dir}")
        
        logger.info(f"✅ DeepSeek OCR 转换成功: {expected_md_file}")
        return str(expected_md_file)
        
    except Exception as e:
        logger.error(f"❌ DeepSeek OCR 转换失败: {str(e)}", exc_info=True)
        raise


def run_command(command, env, step_name):
    """运行命令并实时输出日志"""
    try:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in iter(process.stdout.readline, ''):
            if line:
                logger.info(f"{step_name}: {line.strip()}")
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        logger.info(f"✅ {step_name} 成功，返回码: {return_code}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {step_name} 失败，返回码: {e.returncode}")
        raise

