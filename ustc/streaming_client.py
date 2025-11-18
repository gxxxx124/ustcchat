#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的流式输出客户端
专注于拼接所有JSON块内容并识别停止信号
"""

import asyncio
import aiohttp
import json
import sys

async def stream_chat(url: str, json_data: dict):
    """
    简化的流式聊天函数
    
    Args:
        url: API端点URL
        json_data: 请求的JSON数据
    """
    print(f"🚀 开始流式请求到: {url}")
    print(f"📝 请求内容: {json_data.get('message', 'N/A')}")
    print("-" * 60)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=json_data) as response:
                if response.status != 200:
                    print(f"❌ 请求失败: {response.status}")
                    return
                
                print("📡 开始接收流式数据...")
                
                # 收集所有内容
                all_content = ""
                chunk_count = 0
                stop_signal_received = False
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if not line or not line.startswith('data: '):
                        continue
                    
                    chunk_count += 1
                    data_str = line[6:]  # 移除 'data: ' 前缀
                    
                    try:
                        data = json.loads(data_str)
                        finish_reason = data.get('finish_reason')
                        text = data.get('text', '')
                        
                        # 拼接所有文本内容
                        if text:
                            all_content += text
                        
                        # 检查停止信号
                        if finish_reason == 'stop':
                            stop_signal_received = True
                            print(f"🏁 收到停止信号 (块 {chunk_count})")
                            break
                        elif finish_reason == 'error':
                            print(f"❌ 收到错误信号: {text}")
                            break
                        elif finish_reason == 'final_answer':
                            print(f"📤 收到最终回答信号 (块 {chunk_count})")
                        else:
                            # 普通内容块，显示进度
                            if chunk_count % 10 == 0:  # 每10块显示一次进度
                                print(f"📊 已接收 {chunk_count} 块，当前内容长度: {len(all_content)}")
                        
                    except json.JSONDecodeError:
                        print(f"⚠️ JSON解析失败: {data_str[:50]}...")
                    except Exception as e:
                        print(f"❌ 处理数据失败: {e}")
                
                print("-" * 60)
                print(f"📊 接收完成:")
                print(f"📝 总数据块数: {chunk_count}")
                print(f"📝 总内容长度: {len(all_content)}")
                print(f"🏁 停止信号: {'是' if stop_signal_received else '否'}")
                print("=" * 60)
                
                # 显示完整内容
                if all_content:
                    print("📝 完整内容:")
                    print("-" * 60)
                    print(all_content)
                    print("-" * 60)
                else:
                    print("❌ 没有接收到任何内容")
                
    except Exception as e:
        print(f"❌ 请求异常: {e}")

def main():
    # 测试数据
    json_data = {
        "message": "帮我总结一下报告？",
        "knowledge_base_name": "test-yangaimin",
        "session_id": "test_session12240",
        "enable_web_search": True,
        "url":"chat/7f1c402e-4fea-4002-ac1e-520ecc25d370.pdf"
    }
    
    # 运行流式请求
    asyncio.run(stream_chat('https://7f9b37ca4170.ngrok-free.app/agent/chat/stream', json_data))

if __name__ == "__main__":
    main()
