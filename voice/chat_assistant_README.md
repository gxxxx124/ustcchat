# 智能对话助手

这是一个可以在另一台服务器上运行的智能对话助手，连接到远程数据库存储对话历史，使用相同的 DeepSeek API 接入方式。

## 功能特性

- ✅ 连接到远程 PostgreSQL 数据库存储对话历史
- ✅ 使用 NSRL DeepSeek API（与主服务器相同）
- ✅ **支持工具调用** - 模型可以自动调用工具
- ✅ **RAG知识库MCP工具集成** - 通过HTTP调用MCP知识库服务器进行搜索
- ✅ 支持会话管理（创建、查询、删除）
- ✅ RESTful API 接口
- ✅ 自动保存对话历史

## 安装依赖

```bash
cd voice
pip install -r requirements.txt
```

## 配置

在 `.env` 文件中配置以下环境变量：

```bash
# DeepSeek API 配置（必须）
DEEPSEEK_API_KEY=你的DeepSeek API密钥
DEEPSEEK_API_BASE=http://scc.ustc.edu.cn/portal/api/ask
DEEPSEEK_MODEL=deepseek-v3

# 远程数据库配置（必须）
# 格式: postgresql://user:password@host:port/database?sslmode=disable
CHAT_HISTORY_DB_URI=postgresql://chat_history_user:chat_history_pass@114.214.168.134:5432/chat_history_db?sslmode=disable

# 服务器配置（可选）
CHAT_SERVER_HOST=0.0.0.0
CHAT_SERVER_PORT=8002

# MCP 知识库服务器配置（用于工具调用）
MCP_KNOWLEDGE_SERVER_URL=http://114.214.168.134:8001
```

**重要**: `CHAT_HISTORY_DB_URI` 需要指向主服务器的数据库地址（114.214.168.134）。

## 启动服务

```bash
cd voice
python chat_assistant.py
```

或使用启动脚本：

```bash
./start_chat_assistant.sh
```

## API 端点

### 1. 健康检查

```
GET /health
```

### 2. 智能对话

```
POST /api/chat
```

**请求示例：**
```json
{
  "message": "你好，请介绍一下你自己",
  "session_id": "session_abc123",  // 可选，不提供则创建新会话
  "user_id": 1,  // 可选，默认 1
  "system_prompt": "你是一个专业的AI助手"  // 可选
}
```

**响应示例：**
```json
{
  "session_id": "session_abc123",
  "message": "你好！我是一个AI助手...",
  "conversation_history": [
    {"role": "user", "content": "你好，请介绍一下你自己"},
    {"role": "assistant", "content": "你好！我是一个AI助手..."}
  ]
}
```

### 3. 获取会话列表

```
GET /api/sessions?user_id=1&limit=20
```

### 4. 获取特定会话

```
GET /api/sessions/{session_id}
```

### 5. 删除会话

```
DELETE /api/sessions/{session_id}
```

## 使用示例

### Python 客户端

```python
import requests

BASE_URL = "http://your-server:8002"

# 发送消息
response = requests.post(
    f"{BASE_URL}/api/chat",
    json={
        "message": "如何配置数据库？",
        "session_id": "my_session_123"  # 可选
    }
)

print(response.json())
```

### curl 命令

```bash
# 发送消息
curl -X POST http://your-server:8002/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "如何配置数据库？"
  }'

# 获取会话列表
curl http://your-server:8002/api/sessions?user_id=1

# 获取特定会话
curl http://your-server:8002/api/sessions/session_abc123
```

## 数据库表结构

服务会自动创建以下表（如果不存在）：

### conversations 表
- `conversation_id` (VARCHAR, PRIMARY KEY)
- `user_id` (INTEGER)
- `title` (VARCHAR)
- `message_count` (INTEGER)
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

### messages 表
- `id` (SERIAL, PRIMARY KEY)
- `conversation_id` (VARCHAR, FOREIGN KEY)
- `role` (VARCHAR) - 'user' 或 'assistant'
- `content` (TEXT)
- `created_at` (TIMESTAMP)

## 工具调用功能

智能对话助手支持自动工具调用：

1. **知识库搜索工具**: 当用户询问相关问题时，模型会自动调用知识库搜索工具
2. **自动执行**: 工具调用后，模型会基于搜索结果生成最终回答
3. **MCP集成**: 通过HTTP调用MCP知识库服务器（`http://114.214.168.134:8001`）

### 工具调用流程

1. 用户发送消息
2. 模型判断是否需要调用工具
3. 如果需要，模型调用 `knowledge_base_search` 工具
4. 工具通过MCP服务器搜索知识库
5. 工具返回搜索结果
6. 模型基于搜索结果生成最终回答

## 注意事项

1. **数据库连接**: 确保可以访问主服务器的 PostgreSQL 数据库
2. **网络**: 确保服务器可以访问 DeepSeek API 和 MCP 知识库服务器
3. **防火墙**: 如果需要公网访问，开放相应端口
4. **安全性**: 生产环境建议添加认证和 HTTPS
5. **MCP服务器**: 确保MCP知识库服务器（8001端口）正在运行

## 故障排除

1. **数据库连接失败**: 检查 `CHAT_HISTORY_DB_URI` 配置和网络连接
2. **DeepSeek API 失败**: 检查 `DEEPSEEK_API_KEY` 配置
3. **导入错误**: 确保 `nsrl_deepseek_client.py` 在 Python 路径中

