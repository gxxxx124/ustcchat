# 远程测试指南

## 服务器信息

**服务器地址**: 请替换为实际的公网 IP 地址
**端口**: 8001

## 在别的机器上测试

### 方法一：使用 curl 命令

#### 1. 健康检查
```bash
curl http://YOUR_SERVER_IP:8001/health
```

#### 2. 列出工具
```bash
curl http://YOUR_SERVER_IP:8001/mcp/tools
```

#### 3. 调用 MCP 工具
```bash
curl -X POST http://YOUR_SERVER_IP:8001/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "knowledge_base_search",
    "arguments": {
      "query": "如何配置数据库",
      "collection_name": "nsrl_tech_docs",
      "k": 5
    }
  }'
```

#### 4. 使用 RESTful API
```bash
curl -X POST http://YOUR_SERVER_IP:8001/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "如何配置数据库",
    "k": 5
  }'
```

### 方法二：使用 Python 脚本

```python
import requests

BASE_URL = "http://YOUR_SERVER_IP:8001"

# 健康检查
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 列出工具
response = requests.get(f"{BASE_URL}/mcp/tools")
print(response.json())

# 调用工具
response = requests.post(
    f"{BASE_URL}/mcp/tools/call",
    json={
        "name": "knowledge_base_search",
        "arguments": {
            "query": "如何配置数据库",
            "collection_name": "nsrl_tech_docs",
            "k": 5
        }
    }
)
print(response.json())

# RESTful API
response = requests.post(
    f"{BASE_URL}/api/search",
    json={
        "query": "如何配置数据库",
        "k": 5
    }
)
print(response.json())
```

### 方法三：使用测试脚本

将 `test_remote.sh` 复制到测试机器，然后运行：

```bash
chmod +x test_remote.sh
./test_remote.sh YOUR_SERVER_IP 8001 "如何配置数据库"
```

### 方法四：使用浏览器

直接在浏览器中访问：
- 健康检查: `http://YOUR_SERVER_IP:8001/health`
- 列出工具: `http://YOUR_SERVER_IP:8001/mcp/tools`

## 注意事项

1. **防火墙**: 确保服务器防火墙开放了 8001 端口
2. **安全**: 生产环境建议添加认证和 HTTPS
3. **网络**: 确保测试机器可以访问服务器的公网 IP

## 防火墙配置

如果无法连接，可能需要配置防火墙：

```bash
# Ubuntu/Debian
sudo ufw allow 8001/tcp

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8001/tcp
sudo firewall-cmd --reload
```
