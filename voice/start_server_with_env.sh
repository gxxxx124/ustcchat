#!/bin/bash
# 使用 conda 环境启动 MCP 服务器

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 尝试激活 conda 环境
if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source /opt/anaconda3/etc/profile.d/conda.sh
    conda activate langchain
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate langchain
fi

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
echo "Python: $(which python3)"
echo "Qdrant: $QDRANT_HOST:$QDRANT_PORT"
echo "服务器: $MCP_SERVER_HOST:$MCP_SERVER_PORT"

# 启动服务器
python3 mcp_knowledge_server.py
