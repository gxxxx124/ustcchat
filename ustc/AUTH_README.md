# NSRLChat 认证系统使用说明

## 概述

NSRLChat 现在集成了完整的用户认证系统，只有经过授权的用户才能访问聊天功能。管理员可以创建和管理用户账号。

## 功能特性

- 🔐 用户登录/登出
- 👥 用户账号管理
- 🛡️ API访问保护
- 👨‍💼 管理员控制台
- 🔒 JWT令牌认证

## 快速开始

### 1. 启动服务

```bash
# 使用带认证的启动脚本
./start_with_auth.sh
```

或者手动启动：

```bash
# 1. 初始化管理员账号
python3 init_admin.py

# 2. 启动服务
python3 web_memory.py
```

### 2. 访问应用

- **登录页面**: http://localhost:8000/auth/login-page
- **管理员页面**: http://localhost:8000/auth/admin
- **主应用**: http://localhost:8000/

### 3. 默认管理员账号

- **用户名**: `admin`
- **密码**: `admin123`

⚠️ **重要**: 首次登录后请立即修改默认密码！

## 用户管理

### 管理员功能

管理员可以通过管理员页面 (`/auth/admin`) 进行以下操作：

1. **添加新用户**
   - 设置用户名和密码
   - 选择是否为管理员
   - 可选设置邮箱

2. **查看用户列表**
   - 查看所有用户信息
   - 查看用户创建时间和最后登录时间
   - 识别管理员用户

3. **删除用户**
   - 删除不需要的用户账号
   - 不能删除自己的账号

### 普通用户功能

普通用户登录后可以：

1. **使用聊天功能**
   - 与AI助手对话
   - 访问知识库
   - 上传和管理文档

2. **管理个人对话历史**
   - 查看历史对话
   - 创建新对话
   - 删除对话记录

## API认证

所有API端点现在都需要认证：

### 认证方式

1. **Bearer Token** (推荐)
   ```bash
   curl -H "Authorization: Bearer <your_token>" http://localhost:8000/agent/chat
   ```

2. **Cookie** (用于Web页面)
   - 登录后自动设置 `access_token` cookie

### 受保护的端点

- `/agent/*` - 聊天相关API
- `/kb/*` - 知识库管理API
- `/` - 主页面

### 公开端点

- `/auth/login` - 用户登录
- `/auth/login-page` - 登录页面
- `/auth/admin` - 管理员页面
- `/health` - 健康检查
- `/static/*` - 静态文件

## 数据库结构

认证系统使用PostgreSQL数据库存储用户信息：

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100),
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);
```

## 安全配置

### JWT配置

- **密钥**: `your-secret-key-change-this-in-production`
- **算法**: HS256
- **过期时间**: 30分钟

⚠️ **生产环境安全建议**:

1. 修改JWT密钥
2. 使用环境变量存储敏感配置
3. 启用HTTPS
4. 定期更新密码

### 密码安全

- 使用SHA-256加密存储密码
- 建议用户使用强密码
- 定期提醒用户更新密码

## 故障排除

### 常见问题

1. **无法登录**
   - 检查用户名和密码是否正确
   - 确认用户账号是否存在
   - 查看服务器日志

2. **权限不足**
   - 确认用户是否有管理员权限
   - 检查JWT令牌是否有效

3. **数据库连接失败**
   - 确认PostgreSQL服务正在运行
   - 检查数据库连接配置

### 日志查看

```bash
# 查看应用日志
tail -f logs/app.log

# 查看聊天日志
tail -f logs/chat_flow.log
```

## 开发说明

### 添加新的受保护端点

```python
from auth import get_current_user

@app.get("/protected-endpoint")
async def protected_endpoint(current_user: UserResponse = Depends(get_current_user)):
    return {"message": f"Hello {current_user.username}!"}
```

### 管理员专用端点

```python
from auth import get_current_admin_user

@app.get("/admin-only")
async def admin_only(current_user: UserResponse = Depends(get_current_admin_user)):
    return {"message": "Admin only content"}
```

## 更新日志

- **v1.0.0**: 初始认证系统实现
  - 用户登录/登出
  - 管理员控制台
  - API访问保护
  - JWT令牌认证

## 支持

如有问题，请查看日志文件或联系开发团队。
