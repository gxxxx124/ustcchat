# 子路径部署分析：https://nsrloa.ustc.edu.cn/nsrlchat/

## 当前配置状态

### ✅ 已配置的内容

1. **FastAPI root_path**
   - 在 `web_memory.py` 中已设置：`root_path="/nsrlchat"`
   - 这意味着所有路由会自动添加 `/nsrlchat` 前缀

2. **路径处理函数**
   - `getBasePath()` 函数已实现，支持检测子路径
   - 在多个地方使用，自动处理路径

3. **OAuth回调地址**
   - 使用 `BASE_URL` 环境变量构建回调地址
   - 代码会自动处理子路径

## 部署后的路由映射

当应用挂载在 `https://nsrloa.ustc.edu.cn/nsrlchat/` 下时：

| 原始路由 | 实际访问路径 | 说明 |
|---------|------------|------|
| `/` | `https://nsrloa.ustc.edu.cn/nsrlchat/` | 欢迎页面 ✅ |
| `/chat` | `https://nsrloa.ustc.edu.cn/nsrlchat/chat` | 主应用页面 ✅ |
| `/auth/ustc/login` | `https://nsrloa.ustc.edu.cn/nsrlchat/auth/ustc/login` | USTC登录 ✅ |
| `/auth/ustc/callback` | `https://nsrloa.ustc.edu.cn/nsrlchat/auth/ustc/callback` | OAuth回调 ✅ |
| `/auth/login-page` | `https://nsrloa.ustc.edu.cn/nsrlchat/auth/login-page` | 登录页面 ✅ |
| `/auth/admin` | `https://nsrloa.ustc.edu.cn/nsrlchat/auth/admin` | 管理员页面 ✅ |
| `/kb/*` | `https://nsrloa.ustc.edu.cn/nsrlchat/kb/*` | 知识库API ✅ |
| `/agent/*` | `https://nsrloa.ustc.edu.cn/nsrlchat/agent/*` | Agent API ✅ |
| `/static/*` | `https://nsrloa.ustc.edu.cn/nsrlchat/static/*` | 静态文件 ✅ |

## 需要检查的问题

### 1. ✅ OAuth回调地址配置

**当前状态**：已支持
- `BASE_URL` 环境变量需要设置为：`https://nsrloa.ustc.edu.cn/nsrlchat`
- 回调地址会自动构建为：`https://nsrloa.ustc.edu.cn/nsrlchat/auth/ustc/callback`

**需要做的**：
- 在USTC申请时，回调地址填写：`https://nsrloa.ustc.edu.cn/nsrlchat/auth/ustc/callback`
- 设置环境变量：`export BASE_URL="https://nsrloa.ustc.edu.cn/nsrlchat"`

### 2. ⚠️ 静态文件路径

**当前状态**：可能需要调整
- 静态文件挂载在 `/static`
- HTML文件中的静态资源路径需要检查

**需要检查**：
- `index.html` 中的资源路径（如CSS、JS、图片）
- `welcome.html` 中的资源路径
- `upload.html` 中的资源路径

### 3. ✅ 前端路径处理

**当前状态**：已支持
- `getBasePath()` 函数会自动检测 `/nsrlchat` 前缀
- API调用会使用正确的基础路径

### 4. ✅ Nginx配置

**需要的Nginx配置示例**：

```nginx
location /nsrlchat/ {
    proxy_pass http://localhost:8000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    
    # 重要：设置root_path头部，让FastAPI知道子路径
    proxy_set_header X-Forwarded-Prefix /nsrlchat;
    
    # WebSocket支持（如果需要）
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

## 部署后的完整流程

### 1. 用户访问欢迎页面

用户访问：`https://nsrloa.ustc.edu.cn/nsrlchat/`
- ✅ 显示欢迎页面
- ✅ 所有路径自动添加 `/nsrlchat` 前缀

### 2. 点击登录按钮

跳转到：`https://nsrloa.ustc.edu.cn/nsrlchat/auth/ustc/login`
- ✅ 正确重定向到USTC统一身份认证
- ✅ 回调地址设置为：`https://nsrloa.ustc.edu.cn/nsrlchat/auth/ustc/callback`

### 3. OAuth回调

USTC重定向到：`https://nsrloa.ustc.edu.cn/nsrlchat/auth/ustc/callback?code=...`
- ✅ 正确处理回调
- ✅ 根据GID判断管理员
- ✅ 跳转到正确的页面：
  - 管理员 → `/nsrlchat/auth/admin`
  - 普通用户 → `/nsrlchat/chat`

## 可能遇到的问题

### ❌ 问题1：静态资源404

**症状**：页面显示但CSS/JS/图片加载失败

**原因**：静态资源路径不正确

**解决**：
- 检查HTML中的资源路径
- 确保使用相对路径或正确的绝对路径
- 检查Nginx配置中的静态文件代理

### ❌ 问题2：API请求404

**症状**：前端可以加载但API调用失败

**原因**：前端路径处理不正确

**解决**：
- 检查 `getBasePath()` 函数是否正确工作
- 确认所有API调用都使用了基础路径
- 检查浏览器控制台的网络请求

### ❌ 问题3：OAuth回调失败

**症状**：登录后无法回调

**原因**：回调地址不匹配

**解决**：
- 确认USTC申请的回调地址完全匹配
- 检查 `BASE_URL` 环境变量
- 查看应用日志中的回调地址

### ❌ 问题4：Cookie路径问题

**症状**：登录后无法保持会话

**原因**：Cookie路径设置不正确

**解决**：
- 确保Cookie的path设置为 `/nsrlchat` 或 `/`
- 检查Cookie的domain设置

## 部署检查清单

### 部署前

- [ ] 设置环境变量 `BASE_URL="https://nsrloa.ustc.edu.cn/nsrlchat"`
- [ ] 设置环境变量 `USTC_CLIENT_ID` 和 `USTC_CLIENT_SECRET`
- [ ] 设置环境变量 `ADMIN_GIDS`（如果需要）
- [ ] 在USTC申请时填写正确的回调地址
- [ ] 确认Nginx配置正确

### 部署后测试

- [ ] 访问 `https://nsrloa.ustc.edu.cn/nsrlchat/` 显示欢迎页面
- [ ] 点击登录按钮跳转到USTC
- [ ] USTC登录后正确回调
- [ ] 管理员正确跳转到管理页面
- [ ] 普通用户正确跳转到聊天页面
- [ ] 静态资源正常加载
- [ ] API请求正常
- [ ] 会话保持正常

## 推荐的Nginx配置

```nginx
server {
    listen 443 ssl http2;
    server_name nsrloa.ustc.edu.cn;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    # 子路径代理
    location /nsrlchat/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Prefix /nsrlchat;
        
        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # WebSocket支持
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # 健康检查（可选）
    location /nsrlchat/health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }
}
```

## 总结

✅ **好消息**：代码已经支持子路径部署，主要的路由和路径处理都已实现。

⚠️ **需要注意**：
1. 确保 `BASE_URL` 环境变量正确设置
2. USTC申请时的回调地址必须完全匹配
3. Nginx配置需要正确处理子路径
4. 静态资源路径可能需要调整

🔧 **建议测试步骤**：
1. 先测试欢迎页面是否可以访问
2. 测试登录流程是否完整
3. 测试管理员和普通用户的跳转
4. 检查所有静态资源是否正常加载

