#!/bin/bash
# MCP 知识库搜索服务器启动脚本

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 加载环境变量
if [ -f "../.env" ]; then
    export $(cat ../.env | grep -v '^#' | xargs)
fi

if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# 设置默认值
export QDRANT_HOST=${QDRANT_HOST:-localhost}
export QDRANT_PORT=${QDRANT_PORT:-6333}
export MCP_SERVER_HOST=${MCP_SERVER_HOST:-0.0.0.0}
export MCP_SERVER_PORT=${MCP_SERVER_PORT:-8001}

echo "启动 MCP 知识库搜索服务器..."
echo "Qdrant: $QDRANT_HOST:$QDRANT_PORT"
echo "服务器: $MCP_SERVER_HOST:$MCP_SERVER_PORT"

# 启动服务器
python mcp_knowledge_server.py


