#!/usr/bin/env python3
"""
GPU显存监控脚本
实时监控GPU显存使用情况，帮助诊断显存泄漏问题
"""

import time
import subprocess
import json
from datetime import datetime
import argparse

def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        
        for line in lines:
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 6:
                    gpu_info.append({
                        'index': parts[0],
                        'name': parts[1],
                        'memory_total': int(parts[2]),
                        'memory_used': int(parts[3]),
                        'memory_free': int(parts[4]),
                        'utilization': int(parts[5])
                    })
        
        return gpu_info
    except Exception as e:
        print(f"❌ 获取GPU信息失败: {e}")
        return []

def format_memory(mb):
    """格式化内存大小"""
    if mb >= 1024:
        return f"{mb/1024:.1f} GiB"
    return f"{mb} MiB"

def monitor_gpu(interval=5, alert_threshold=80):
    """监控GPU显存使用情况"""
    print(f"🔍 GPU显存监控启动 (检查间隔: {interval}秒, 告警阈值: {alert_threshold}%)")
    print("=" * 80)
    
    try:
        while True:
            gpu_info = get_gpu_info()
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n⏰ {current_time}")
            print("-" * 60)
            
            for gpu in gpu_info:
                memory_usage_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
                status = "🟢" if memory_usage_percent < alert_threshold else "🔴"
                
                print(f"{status} GPU {gpu['index']}: {gpu['name']}")
                print(f"   显存: {format_memory(gpu['memory_used'])} / {format_memory(gpu['memory_total'])} ({memory_usage_percent:.1f}%)")
                print(f"   可用: {format_memory(gpu['memory_free'])} | 利用率: {gpu['utilization']}%")
                
                if memory_usage_percent >= alert_threshold:
                    print(f"   ⚠️  显存使用率过高！")
                
                print()
            
            # 检查是否有进程占用大量显存
            try:
                result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, check=True)
                if result.stdout.strip():
                    print("📊 显存占用进程:")
                    print(result.stdout.strip())
                    print()
            except:
                pass
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 监控已停止")
    except Exception as e:
        print(f"\n❌ 监控出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='GPU显存监控工具')
    parser.add_argument('-i', '--interval', type=int, default=5, help='检查间隔(秒), 默认5秒')
    parser.add_argument('-t', '--threshold', type=int, default=80, help='告警阈值(%), 默认80%')
    
    args = parser.parse_args()
    
    # 检查nvidia-smi是否可用
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, check=True)
    except:
        print("❌ nvidia-smi不可用，请确保已安装NVIDIA驱动")
        return
    
    monitor_gpu(args.interval, args.threshold)

if __name__ == "__main__":
    main()
