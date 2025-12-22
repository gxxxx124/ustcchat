#!/bin/bash
# 智能对话助手启动脚本

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
export DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:-""}
export DEEPSEEK_API_BASE=${DEEPSEEK_API_BASE:-"http://scc.ustc.edu.cn/portal/api/ask"}
export DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-"deepseek-v3"}
export CHAT_SERVER_HOST=${CHAT_SERVER_HOST:-"0.0.0.0"}
export CHAT_SERVER_PORT=${CHAT_SERVER_PORT:-"8002"}

echo "启动智能对话助手..."
echo "Python: $(which python3)"
echo "服务器: $CHAT_SERVER_HOST:$CHAT_SERVER_PORT"
echo "DeepSeek API: $DEEPSEEK_API_BASE"

# 启动服务器
python3 chat_assistant.py


