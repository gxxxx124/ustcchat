#!/usr/bin/env python3
"""
简化的聊天测试，直接使用DeepSeek API
"""
import os
import asyncio
from openai import AsyncOpenAI

async def simple_chat():
    """简单的聊天测试"""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("❌ 错误: 请设置环境变量 DEEPSEEK_API_KEY")
        return False
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    
    try:
        print("🔍 测试DeepSeek API...")
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个有用的助手"},
                {"role": "user", "content": "你好，请简单介绍一下你自己"}
            ],
            max_tokens=200,
            timeout=30.0
        )
        
        print("✅ API调用成功!")
        print(f"📝 回复: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ API调用失败: {str(e)}")
        return False

if __name__ == "__main__":
    result = asyncio.run(simple_chat())
    print(f"\n测试结果: {'成功' if result else '失败'}")
