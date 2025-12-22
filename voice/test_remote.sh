#!/bin/bash
# 远程测试脚本 - 在别的机器上运行此脚本

# 配置服务器地址（替换为实际的公网 IP）
SERVER_IP="${1:-your-server-ip}"
SERVER_PORT="${2:-8001}"
BASE_URL="http://${SERVER_IP}:${SERVER_PORT}"

echo "=========================================="
echo "MCP 知识库搜索服务器远程测试"
echo "服务器地址: ${BASE_URL}"
echo "=========================================="
echo ""

# 1. 测试健康检查
echo "1. 测试健康检查..."
curl -s "${BASE_URL}/health" | python3 -m json.tool 2>/dev/null || curl -s "${BASE_URL}/health"
echo ""
echo ""

# 2. 测试列出工具
echo "2. 测试列出工具..."
curl -s "${BASE_URL}/mcp/tools" | python3 -m json.tool 2>/dev/null || curl -s "${BASE_URL}/mcp/tools"
echo ""
echo ""

# 3. 测试调用工具
echo "3. 测试调用工具..."
QUERY="${3:-如何配置数据库}"
curl -s -X POST "${BASE_URL}/mcp/tools/call" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"knowledge_base_search\",
    \"arguments\": {
      \"query\": \"${QUERY}\",
      \"collection_name\": \"nsrl_tech_docs\",
      \"k\": 5
    }
  }" | python3 -m json.tool 2>/dev/null || curl -s -X POST "${BASE_URL}/mcp/tools/call" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"knowledge_base_search\",
    \"arguments\": {
      \"query\": \"${QUERY}\",
      \"collection_name\": \"nsrl_tech_docs\",
      \"k\": 5
    }
  }"
echo ""
echo ""

# 4. 测试 RESTful API
echo "4. 测试 RESTful API..."
curl -s -X POST "${BASE_URL}/api/search" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"${QUERY}\",
    \"k\": 5
  }" | python3 -m json.tool 2>/dev/null || curl -s -X POST "${BASE_URL}/api/search" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"${QUERY}\",
    \"k\": 5
  }"
echo ""
echo ""

echo "=========================================="
echo "测试完成！"
echo "=========================================="
