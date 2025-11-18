# API 切换指南

## 当前配置

当前 `web_memory.py` 使用的是 **Open WebUI API**（本地部署）

## 文件说明

- `web_memory.py` - 主服务文件（当前使用 Open WebUI API）
- `web_api.py` - DeepSeek API 配置说明
- `web_api_deepseek_config.py` - DeepSeek API 配置代码片段
- `web_api_deepseek_backup.py` - DeepSeek API 完整配置备份

## 切换到 DeepSeek API

### 步骤 1: 修改 get_ollama_model() 方法（第 205-229 行）

**当前代码（Open WebUI）**：
```python
def get_ollama_model(self):
    """获取Open WebUI API模型实例（按需加载）"""
    if "ollama" not in self.model_instances:
        api_key = os.getenv("OPENWEBUI_API_KEY", "")  # 必须从环境变量设置
        api_base = os.getenv("OPENWEBUI_API_BASE", "")  # 必须从环境变量设置
        model_name = os.getenv("OPENWEBUI_MODEL", "../../models/deepseek-r1-70b/deepseek-r1-70b.gguf")
        # ...
```

**替换为（DeepSeek）**：
```python
def get_ollama_model(self):
    """获取DeepSeek API模型实例（按需加载）"""
    if "ollama" not in self.model_instances:
        api_key = os.getenv("DEEPSEEK_API_KEY", "")  # 必须从环境变量设置
        api_base = "https://api.deepseek.com"
        model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        # ...
```

### 步骤 2: 修改 call_model() 方法（第 2003-2010 行）

**当前代码（Open WebUI）**：
```python
# Open WebUI 不支持 tools 参数，直接使用模型（工具调用通过文本解析实现）
model = gpu_resource_manager.get_ollama_model()
chat_logger.info(f"🤖 获取到模型类型: {type(model).__name__}")
chat_logger.info(f"ℹ️ Open WebUI 不支持标准工具绑定，将使用文本解析方式处理工具调用")

chat_logger.info(f"🤖 开始调用Open WebUI API模型...")
# 直接调用模型（不绑定工具，工具调用信息已在 system_prompt 中描述）
response = await model.ainvoke(messages)
```

**替换为（DeepSeek）**：
```python
# 始终绑定所有可用工具
model = gpu_resource_manager.get_ollama_model()
chat_logger.info(f"🤖 获取到模型类型: {type(model).__name__}")
model_with_tools = model.bind_tools(available_tools)

chat_logger.info(f"🤖 开始调用DeepSeek API模型...")
# 调用模型
response = await model_with_tools.ainvoke(messages)
```

### 步骤 3: 重启服务

```bash
# 停止服务
ps aux | grep web_memory | grep -v grep | awk '{print $2}' | xargs kill

# 等待服务停止
sleep 2

# 启动服务
cd /home/user/ustcchat
conda activate langchain
nohup python ustc/web_memory.py > server.log 2>&1 &

# 查看日志确认
tail -f server.log
```

## 切换回 Open WebUI API

执行相反的操作，将 DeepSeek 配置替换回 Open WebUI 配置。

## 配置对比

| 特性 | DeepSeek API | Open WebUI API |
|------|-------------|----------------|
| API 地址 | `https://api.deepseek.com` | 从环境变量 `OPENWEBUI_API_BASE` 读取 |
| 工具绑定 | 支持 `bind_tools()` | 不支持，使用文本解析 |
| 工具调用方式 | 标准 `tool_calls` 属性 | 文本解析 JSON |
| 模型名称 | `deepseek-chat` | `../../models/deepseek-r1-70b/deepseek-r1-70b.gguf` |

## 注意事项

1. **DeepSeek API**：
   - 支持标准的 `tools` 参数
   - 可以使用 `model.bind_tools(available_tools)`
   - 模型返回标准的 `tool_calls` 属性

2. **Open WebUI API**：
   - 不支持 `tools` 参数（需要 `--jinja` 标志）
   - 不能使用 `bind_tools()`
   - 工具调用通过文本解析实现（备用方案）

3. **备份文件**：
   - 所有配置都已备份在 `web_api*.py` 文件中
   - 可以随时参考或恢复

