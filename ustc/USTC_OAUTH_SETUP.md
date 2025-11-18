# USTC 统一身份认证对接说明

本文档说明如何将 NSRLChat 登录系统接入中科大统一身份认证系统。

## 1. 对接流程概述

USTC统一身份认证系统支持 OAuth2.0 授权码模式，对接流程包括：
1. 用户点击"使用中科大统一身份认证登录"按钮
2. 重定向到 USTC 统一身份认证页面
3. 用户在 USTC 页面输入账号密码登录
4. USTC 重定向回应用，携带授权码 (code)
5. 应用使用授权码换取 access_token
6. 应用使用 access_token 获取用户信息
7. 将用户信息同步到本地数据库，生成本地 JWT token

## 2. 前置条件

### 2.1 申请接入

根据 USTC 开发者文档 (https://id.ustc.edu.cn/doc/developer/)，需要：

1. **网页应用**：在[网络安全工作平台](https://netsecurity.ustc.edu.cn/)完成建站申请及网站备案后，由网站负责人或管理员填写[统一身份认证接入申请](https://service.ustc.edu.cn/fe/taskCenter/one/application?app_id=234)

2. **其他应用类型**：联系 wf0229@ustc.edu.cn

系统管理员收到申请后会提供：
- `client_id`：应用的 Client ID
- `client_secret`：应用的 Client Secret

### 2.2 配置回调地址

在申请时，需要提供回调地址（Redirect URI），格式为：
```
https://your-domain.com/auth/ustc/callback
```

**重要**：回调地址必须与代码中配置的完全一致。

## 3. 环境变量配置

在主应用启动前，需要设置以下环境变量：

```bash
export USTC_CLIENT_ID="your_client_id_here"
export USTC_CLIENT_SECRET="your_client_secret_here"
```

或者在 `.env` 文件中配置：
```
USTC_CLIENT_ID=your_client_id_here
USTC_CLIENT_SECRET=your_client_secret_here
```

## 4. 代码修改

### 4.1 在主应用中初始化 USTC OAuth

在 `web_memory.py` 或其他主应用文件中，添加：

```python
from auth_routes import init_ustc_oauth

# 获取应用的基础URL
BASE_URL = "https://your-domain.com"  # 或从配置中读取

# 初始化USTC OAuth
init_ustc_oauth(BASE_URL)
```

### 4.2 确保已安装依赖

确保已安装 `httpx` 库：
```bash
pip install httpx
```

## 5. 使用说明

### 5.1 用户登录流程

1. 用户访问登录页面：`/auth/login-page`
2. 可以选择两种登录方式：
   - **传统登录**：输入用户名和密码
   - **USTC统一身份认证**：点击"使用中科大统一身份认证登录"按钮

### 5.2 首次使用USTC登录

- 系统会自动在本地数据库中创建用户账号
- 用户名使用 USTC 的 GID
- 邮箱使用 USTC 提供的邮箱地址
- 默认不是管理员，需要手动设置为管理员

### 5.3 管理员设置

如果USTC登录的用户需要管理员权限，可以通过以下方式设置：

1. 使用现有的管理员账号登录
2. 进入管理员控制台 (`/auth/admin`)
3. 在用户列表中找到对应的用户
4. 通过数据库直接修改，或在代码中添加设置管理员的接口

## 6. 代码文件说明

- `ustc_oauth.py`：USTC OAuth2.0 客户端实现
- `auth_routes.py`：认证路由，包含：
  - `/auth/ustc/login`：USTC登录入口
  - `/auth/ustc/callback`：OAuth回调处理

## 7. 故障排查

### 7.1 USTC登录按钮不可见

检查：
- 环境变量 `USTC_CLIENT_ID` 和 `USTC_CLIENT_SECRET` 是否已设置
- 查看应用日志，确认是否有警告信息

### 7.2 登录后无法跳转

检查：
- 回调地址是否正确配置
- 回调地址是否在 USTC 系统中已注册
- 查看应用日志中的错误信息

### 7.3 获取用户信息失败

检查：
- `access_token` 是否有效（有效期8小时）
- 网络连接是否正常
- USTC 服务器状态

## 8. 安全注意事项

1. **保护 client_secret**：
   - 不要将 `client_secret` 提交到代码仓库
   - 使用环境变量或密钥管理系统存储

2. **state 参数**：
   - 当前实现中简化了 state 参数的处理
   - 生产环境建议使用 session 或 Redis 存储 state，防止 CSRF 攻击

3. **HTTPS**：
   - 生产环境必须使用 HTTPS
   - 否则 OAuth 流程可能失败

4. **回调地址验证**：
   - 确保回调地址配置正确，防止开放重定向漏洞

## 9. 参考文档

- [USTC统一身份认证开发者手册](https://id.ustc.edu.cn/doc/developer/)
- [OAuth 2.0 RFC 6749](https://tools.ietf.org/html/rfc6749)

## 10. 联系支持

如有问题，可以：
1. 查看应用日志
2. 联系 USTC 统一身份认证管理员：wf0229@ustc.edu.cn
3. 参考 USTC 开发者文档

