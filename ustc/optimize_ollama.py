#!/usr/bin/env python3
"""
Ollama显存优化配置脚本
帮助优化Ollama服务的显存使用，防止显存泄漏
"""

import json
import subprocess
import time
import requests
from pathlib import Path
import os

class OllamaOptimizer:
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.config_dir = Path.home() / ".ollama"
        
    def check_ollama_status(self):
        """检查Ollama服务状态"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama服务运行正常")
                return True
            else:
                print(f"⚠️  Ollama服务响应异常: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 无法连接到Ollama服务: {e}")
            return False
    
    def get_loaded_models(self):
        """获取已加载的模型"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                print(f"📋 已加载模型数量: {len(models)}")
                for model in models:
                    print(f"   - {model.get('name', 'Unknown')} ({model.get('size', 0)} bytes)")
                return models
            return []
        except Exception as e:
            print(f"❌ 获取模型列表失败: {e}")
            return []
    
    def unload_unused_models(self):
        """卸载未使用的模型以释放显存"""
        try:
            # 获取正在运行的模型
            response = requests.get(f"{self.ollama_url}/api/ps")
            if response.status_code == 200:
                running_models = response.json().get("models", [])
                print(f"🔄 正在运行的模型: {len(running_models)}")
                
                # 卸载所有模型以释放显存
                for model in running_models:
                    model_name = model.get("name")
                    if model_name:
                        print(f"   🗑️  卸载模型: {model_name}")
                        try:
                            unload_response = requests.post(f"{self.ollama_url}/api/generate", 
                                                         json={"model": model_name, "prompt": "", "stream": False})
                            if unload_response.status_code == 200:
                                print(f"      ✅ 模型 {model_name} 卸载成功")
                            else:
                                print(f"      ⚠️  模型 {model_name} 卸载失败")
                        except Exception as e:
                            print(f"      ❌ 卸载模型 {model_name} 时出错: {e}")
                
                print("🧹 显存清理完成")
                return True
            return False
        except Exception as e:
            print(f"❌ 卸载模型失败: {e}")
            return False
    
    def optimize_model_config(self, model_name="qwen3:4b"):
        """优化模型配置以减少显存占用"""
        config = {
            "model": model_name,
            "options": {
                "num_ctx": 2048,        # 减少上下文长度
                "num_gpu": 1,           # 限制GPU数量
                "num_thread": 4,        # 限制线程数
                "f16": True,            # 使用半精度浮点数
                "low_vram": True,       # 启用低显存模式
                "rope_scaling": {"type": "linear", "factor": 1.0},  # 优化位置编码
                "mirostat": 2,          # 启用mirostat采样
                "mirostat_tau": 5.0,    # 设置mirostat参数
                "mirostat_eta": 0.1
            }
        }
        
        print(f"⚙️  优化模型配置: {model_name}")
        print(f"   上下文长度: {config['options']['num_ctx']}")
        print(f"   低显存模式: {config['options']['low_vram']}")
        print(f"   半精度: {config['options']['f16']}")
        
        return config
    
    def create_optimized_pull_command(self, model_name="qwen3:4b"):
        """创建优化的模型拉取命令"""
        config = self.optimize_model_config(model_name)
        
        # 构建ollama pull命令
        cmd = f"ollama pull {model_name}"
        
        # 创建Modelfile
        modelfile_content = f"""FROM {model_name}
PARAMETER num_ctx {config['options']['num_ctx']}
PARAMETER num_gpu {config['options']['num_gpu']}
PARAMETER num_thread {config['options']['num_thread']}
PARAMETER f16 {str(config['options']['f16']).lower()}
PARAMETER low_vram {str(config['options']['low_vram']).lower()}
PARAMETER rope_scaling {json.dumps(config['options']['rope_scaling'])}
PARAMETER mirostat {config['options']['mirostat']}
PARAMETER mirostat_tau {config['options']['mirostat_tau']}
PARAMETER mirostat_eta {config['options']['mirostat_eta']}
"""
        
        modelfile_path = self.config_dir / f"{model_name}.Modelfile"
        modelfile_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        print(f"📝 已创建优化配置文件: {modelfile_path}")
        print(f"🚀 使用以下命令拉取优化模型:")
        print(f"   ollama create {model_name}-optimized -f {modelfile_path}")
        print(f"   ollama run {model_name}-optimized")
        
        return modelfile_path
    
    def monitor_memory_usage(self, duration=60):
        """监控显存使用情况"""
        print(f"📊 开始监控显存使用情况 ({duration}秒)")
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # 获取GPU信息
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, check=True)
                
                if result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    for i, line in enumerate(lines):
                        if line.strip():
                            parts = line.split(', ')
                            if len(parts) >= 2:
                                used = int(parts[0])
                                free = int(parts[1])
                                total = used + free
                                usage_percent = (used / total) * 100
                                
                                print(f"GPU {i}: {used}MB / {total}MB ({usage_percent:.1f}%) | 可用: {free}MB")
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n🛑 监控已停止")
        except Exception as e:
            print(f"❌ 监控出错: {e}")
    
    def restart_ollama_service(self):
        """重启Ollama服务以清理显存"""
        print("🔄 重启Ollama服务...")
        
        try:
            # 停止Ollama服务
            subprocess.run(['pkill', '-f', 'ollama'], check=False)
            time.sleep(2)
            
            # 启动Ollama服务
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            print("⏳ 等待服务启动...")
            time.sleep(10)
            
            # 检查服务状态
            if self.check_ollama_status():
                print("✅ Ollama服务重启成功")
                return True
            else:
                print("❌ Ollama服务重启失败")
                return False
                
        except Exception as e:
            print(f"❌ 重启服务失败: {e}")
            return False

def main():
    print("🚀 Ollama显存优化工具")
    print("=" * 50)
    
    optimizer = OllamaOptimizer()
    
    # 检查服务状态
    if not optimizer.check_ollama_status():
        print("❌ Ollama服务未运行，请先启动服务")
        return
    
    while True:
        print("\n📋 请选择操作:")
        print("1. 检查当前模型状态")
        print("2. 卸载未使用模型")
        print("3. 创建优化配置")
        print("4. 监控显存使用")
        print("5. 重启Ollama服务")
        print("6. 退出")
        
        choice = input("\n请输入选择 (1-6): ").strip()
        
        if choice == "1":
            optimizer.get_loaded_models()
        elif choice == "2":
            optimizer.unload_unused_models()
        elif choice == "3":
            model_name = input("请输入模型名称 (默认: qwen3:4b): ").strip() or "qwen3:4b"
            optimizer.create_optimized_pull_command(model_name)
        elif choice == "4":
            duration = input("请输入监控时长(秒, 默认60): ").strip()
            duration = int(duration) if duration.isdigit() else 60
            optimizer.monitor_memory_usage(duration)
        elif choice == "5":
            optimizer.restart_ollama_service()
        elif choice == "6":
            print("👋 再见!")
            break
        else:
            print("❌ 无效选择，请重新输入")

if __name__ == "__main__":
    main()
