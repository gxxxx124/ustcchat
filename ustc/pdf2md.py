import os
import subprocess


def pdf2md(pdf_path):
    # 1. 手动设置关键环境变量（模拟 conda activate）
    env = os.environ.copy()
    env["PATH"] = "/opt/anaconda3/envs/ocrflux/bin:" + env["PATH"]  # 关键！
    env["CONDA_PREFIX"] = "/opt/anaconda3/envs/ocrflux"

    # 2. 第一步：PDF转JSONL
    pipeline_cmd = [
        "/opt/anaconda3/envs/ocrflux/bin/python3.11",
        "-m", "ocrflux.pipeline",
        "/home/easyai/OCRFlux/localworkspace",
        "--data", pdf_path,
        "--model", "/home/easyai/OCRFlux/models/OCRFlux-3B"
    ]
    run_command(pipeline_cmd, env, "pdf2json")

    # 3. 第二步：JSONL转Markdown
    md_cmd = [
        "/opt/anaconda3/envs/ocrflux/bin/python3.11",
        "-m", "ocrflux.jsonl_to_markdown",
        "/home/easyai/OCRFlux/localworkspace"
    ]
    run_command(md_cmd, env, "pdf2md")


def run_command(command, env, step_name):
    try:
        process = subprocess.Popen(
            command,
            env=env,  # 传入定制环境变量
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        print(f"{step_name}成功，返回码: {return_code}")
    except subprocess.CalledProcessError as e:
        print(f"{step_name}失败，返回码: {e.returncode}")
        raise  # 可选：终止后续流程


