# NSRLChat 子路径部署说明

## 问题描述
您之前将NSRLChat部署在 `nsrloa.ustc.edu.cn/NSRLChat` 路径下，现在访问显示404。

## 解决方案

### 1. 服务器修复
- ✅ 修复了 `Depends` 导入错误
- ✅ 添加了子路径支持 (`root_path="/NSRLChat"`)
- ✅ 服务器现在正常运行在8000端口

### 2. Nginx配置
创建了nginx配置文件 `nginx_nsrlchat.conf`，用于将服务挂载到子路径下。

#### 配置步骤：
1. 将 `nginx_nsrlchat.conf` 复制到nginx配置目录
2. 启用配置：`sudo ln -s /etc/nginx/sites-available/nsrlchat /etc/nginx/sites-enabled/`
3. 测试配置：`sudo nginx -t`
4. 重启nginx：`sudo systemctl restart nginx`

### 3. 当前状态
- ✅ 服务器运行正常：`http://localhost:8000`
- ✅ 子路径支持：`http://localhost:8000/NSRLChat/`
- ✅ 健康检查：`http://localhost:8000/health`

### 4. 访问地址
- 登录页面：`http://nsrloa.ustc.edu.cn/NSRLChat/auth/login-page`
- 管理员页面：`http://nsrloa.ustc.edu.cn/NSRLChat/auth/admin`
- 主应用：`http://nsrloa.ustc.edu.cn/NSRLChat/`
- 文件管理：`http://nsrloa.ustc.edu.cn/NSRLChat/upload.html`

### 5. 默认管理员账号
- 用户名：`admin`
- 密码：`admin123`

### 6. 权限控制
- ✅ 普通用户看不到文件管理按钮
- ✅ 只有管理员才能访问文件管理页面
- ✅ 管理员可以创建其他管理员

## 故障排除

### 如果仍然显示404：
1. 检查nginx是否运行：`sudo systemctl status nginx`
2. 检查nginx配置：`sudo nginx -t`
3. 检查域名解析：`nslookup nsrloa.ustc.edu.cn`
4. 检查防火墙：`sudo ufw status`

### 如果服务无法启动：
1. 检查Python环境：`/home/user/miniconda3/envs/langchain/bin/python --version`
2. 检查PostgreSQL：`pg_isready -h localhost -p 5432`
3. 检查Qdrant：`curl -s http://localhost:6333/collections`

## 测试命令
```bash
# 测试本地服务
curl -s http://localhost:8000/health

# 测试子路径
curl -s -I http://localhost:8000/NSRLChat/

# 测试登录
curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```
