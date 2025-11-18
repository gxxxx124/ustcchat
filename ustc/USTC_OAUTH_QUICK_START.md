# USTC 统一身份认证快速配置指南

## ✅ 已完成的功能

1. **OAuth2.0 客户端实现** (`ustc_oauth.py`)
   - 生成授权URL
   - 使用授权码换取access_token
   - 获取用户信息
   - 解析用户属性

2. **认证路由** (`auth_routes.py`)
   - `/auth/ustc/login` - USTC登录入口
   - `/auth/ustc/callback` - OAuth回调处理
   - 自动同步USTC用户到本地数据库

3. **登录页面更新**
   - 添加"使用中科大统一身份认证登录"按钮
   - 错误提示处理

## 🚀 使用步骤

### 第一步：申请接入

1. 访问 [USTC统一身份认证接入申请](https://service.ustc.edu.cn/fe/taskCenter/one/application?app_id=234)
2. 填写申请表，提供回调地址：
   ```
   https://your-domain.com/auth/ustc/callback
   ```
3. 等待管理员审核，获取 `client_id` 和 `client_secret`

### 第二步：配置环境变量

在主应用启动前，设置环境变量：

```bash
# USTC OAuth配置
export USTC_CLIENT_ID="你的client_id"
export USTC_CLIENT_SECRET="你的client_secret"

# 应用基础URL（可选，默认http://localhost:8000）
export BASE_URL="https://your-domain.com"
```

或在 `.env` 文件中：
```
USTC_CLIENT_ID=你的client_id
USTC_CLIENT_SECRET=你的client_secret
BASE_URL=https://your-domain.com
```

### 第三步：安装依赖

确保已安装 `httpx`：
```bash
pip install httpx
```

### 第四步：启动应用

正常启动应用即可，USTC OAuth会自动初始化。

```bash
python3 web_memory.py
```

如果配置正确，日志中会显示：
```
USTC OAuth初始化成功，回调地址: https://your-domain.com/auth/ustc/callback
```

### 第五步：测试登录

1. 访问登录页面：`http://your-domain.com/auth/login-page`
2. 点击"使用中科大统一身份认证登录"按钮
3. 在USTC页面输入账号密码
4. 登录成功后自动跳转回应用

## 📋 注意事项

1. **回调地址必须完全匹配**
   - 在USTC申请时填写的回调地址
   - 代码中使用的回调地址
   - 两者必须完全一致（包括协议、域名、路径）

2. **首次登录自动创建用户**
   - 系统会自动在本地数据库创建用户
   - 用户名使用USTC的GID
   - 默认不是管理员

3. **HTTPS要求**
   - 生产环境必须使用HTTPS
   - 否则OAuth流程可能失败

4. **环境变量安全**
   - 不要将 `client_secret` 提交到代码仓库
   - 使用环境变量或密钥管理系统

## 🔍 故障排查

### 问题：登录按钮不可见
**原因**：环境变量未设置或为空
**解决**：检查 `USTC_CLIENT_ID` 和 `USTC_CLIENT_SECRET` 是否正确设置

### 问题：点击登录后报错"USTC OAuth未配置"
**原因**：初始化失败
**解决**：查看应用启动日志，确认是否有错误信息

### 问题：回调失败
**原因**：回调地址不匹配
**解决**：确保申请时的回调地址与代码中的完全一致

### 问题：无法获取用户信息
**原因**：网络问题或token过期
**解决**：检查网络连接，查看日志中的详细错误信息

## 📞 获取帮助

- USTC统一身份认证文档：https://id.ustc.edu.cn/doc/developer/
- 联系管理员：wf0229@ustc.edu.cn
- 查看详细配置：`USTC_OAUTH_SETUP.md`

