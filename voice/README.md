# MCP 知识库搜索服务器

这是一个符合 MCP (Model Context Protocol) 规范的知识库搜索服务，可以通过 HTTP API 供公网服务器调用。

## 功能特性

- ✅ 符合 MCP 协议规范
- ✅ 支持语义搜索和混合搜索
- ✅ RESTful API 接口
- ✅ 支持多个知识库集合
- ✅ 可配置的搜索参数（结果数量、权重等）
- ✅ CORS 支持，可跨域访问
- ✅ 健康检查端点

## 安装依赖

```bash
cd voice
pip install -r requirements.txt
```

## 配置

创建 `.env` 文件（或使用项目根目录的 `.env`）：

```bash
# Qdrant 向量数据库配置
QDRANT_HOST=localhost
QDRANT_PORT=6333

# MCP 服务器配置
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=8001
```

## 启动服务

```bash
cd voice
python mcp_knowledge_server.py
```

服务将在 `http://0.0.0.0:8001` 启动。

## API 端点

### 1. 根端点

```
GET /
```

返回服务信息。

### 2. 健康检查

```
GET /health
```

检查服务健康状态。

### 3. MCP 协议端点

#### 列出工具

```
GET /mcp/tools
```

返回可用的工具列表。

**响应示例：**
```json
{
  "tools": [
    {
      "name": "knowledge_base_search",
      "description": "在知识库中搜索相关信息。支持语义搜索和混合搜索。",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "搜索查询语句"
          },
          "collection_name": {
            "type": "string",
            "description": "知识库集合名称，默认为 nsrl_tech_docs",
            "default": "nsrl_tech_docs"
          },
          "k": {
            "type": "integer",
            "description": "返回结果数量，范围 1-50，默认 15",
            "default": 15
          }
        },
        "required": ["query"]
      }
    }
  ]
}
```

#### 调用工具

```
POST /mcp/tools/call
```

调用知识库搜索工具。

**请求示例：**
```json
{
  "name": "knowledge_base_search",
  "arguments": {
    "query": "如何配置数据库",
    "collection_name": "nsrl_tech_docs",
    "k": 10,
    "title_weight": 0.6,
    "content_weight": 0.4
  }
}
```

**响应示例：**
```json
{
  "content": [
    {
      "type": "text",
      "text": "📚 在知识库中找到 10 个相关文档片段：\n【文档片段 - 结果 #1 (相似度: 0.8523)】\n标题: 数据库配置指南\n内容: ...\n..."
    }
  ],
  "isError": false
}
```

### 4. RESTful API 端点

#### 搜索知识库

```
POST /api/search
```

简单的 RESTful API 接口。

**请求示例：**
```json
{
  "query": "如何配置数据库",
  "collection_name": "nsrl_tech_docs",
  "k": 10,
  "title_weight": 0.6,
  "content_weight": 0.4
}
```

**响应示例：**
```json
{
  "success": true,
  "results": [
    {
      "index": 1,
      "score": 0.8523,
      "content": "数据库配置需要...",
      "metadata": {
        "title": "数据库配置指南",
        "source": "docs/database.md",
        "is_qa_pair": false
      }
    }
  ],
  "total": 10
}
```

## 部署到公网

### 使用 uvicorn 直接运行

```bash
uvicorn mcp_knowledge_server:app --host 0.0.0.0 --port 8001
```

### 使用 systemd 服务

创建 `/etc/systemd/system/mcp-knowledge.service`：

```ini
[Unit]
Description=MCP Knowledge Base Search Server
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/ustcchat/voice
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python mcp_knowledge_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl enable mcp-knowledge
sudo systemctl start mcp-knowledge
```

### 使用 Nginx 反向代理

在 Nginx 配置中添加：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### 使用 Docker

创建 `Dockerfile`：

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["python", "mcp_knowledge_server.py"]
```

构建和运行：
```bash
docker build -t mcp-knowledge-server .
docker run -d -p 8001:8001 --env-file .env mcp-knowledge-server
```

## 安全建议

1. **限制 CORS 来源**：在生产环境中，修改 `allow_origins` 为具体的域名列表
2. **添加认证**：如果需要，可以添加 API Key 或 JWT 认证
3. **使用 HTTPS**：通过 Nginx 或反向代理配置 SSL/TLS
4. **限制访问**：使用防火墙规则限制访问来源

## 测试

使用 curl 测试：

```bash
# 健康检查
curl http://localhost:8001/health

# 列出工具
curl http://localhost:8001/mcp/tools

# 调用工具
curl -X POST http://localhost:8001/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "knowledge_base_search",
    "arguments": {
      "query": "如何配置数据库"
    }
  }'

# RESTful API
curl -X POST http://localhost:8001/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何配置数据库",
    "k": 5
  }'
```

## 故障排除

1. **无法连接 Qdrant**：检查 `QDRANT_HOST` 和 `QDRANT_PORT` 配置
2. **导入错误**：确保 `ustc` 模块在 Python 路径中
3. **端口被占用**：修改 `MCP_SERVER_PORT` 环境变量

## 许可证

与主项目相同。


