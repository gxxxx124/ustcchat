# 登录流程说明

## 完整的登录流程

### 1. 欢迎页面（无需认证）

访问：`https://nsrloa.ustc.edu.cn/nsrlchat/`

- 显示欢迎界面
- 有"使用中科大统一身份认证登录"按钮
- 点击按钮跳转到 `/auth/ustc/login`

### 2. USTC统一身份认证

点击登录按钮后：
- 重定向到 USTC 统一身份认证页面
- 用户输入 USTC 账号密码
- USTC 认证成功后重定向回应用

### 3. 回调处理

应用收到回调后：
1. 使用授权码换取 access_token
2. 使用 access_token 获取用户信息（包含GID）
3. 根据GID判断是否为管理员
4. 同步用户到本地数据库
5. 生成本地JWT token

### 4. 跳转逻辑

根据用户角色跳转：
- **管理员** (`is_admin=True`)：跳转到 `/nsrlchat/auth/admin`
- **普通用户** (`is_admin=False`)：跳转到 `/nsrlchat/chat`

## 路由说明

| 路径 | 说明 | 需要认证 |
|------|------|---------|
| `/` | 欢迎页面 | 否 |
| `/chat` | 主应用对话页面 | 是 |
| `/auth/ustc/login` | USTC登录入口 | 否 |
| `/auth/ustc/callback` | OAuth回调处理 | 否 |
| `/auth/admin` | 管理员控制台 | 是（管理员） |

## 环境变量配置

```bash
# USTC OAuth配置
export USTC_CLIENT_ID="你的client_id"
export USTC_CLIENT_SECRET="你的client_secret"

# 应用基础URL
export BASE_URL="https://nsrloa.ustc.edu.cn/nsrlchat"

# 管理员GID列表（用逗号分隔）
export ADMIN_GIDS="9202420483,9202420484"
```

## 管理员判断

系统通过环境变量 `ADMIN_GIDS` 配置管理员GID列表：

1. 用户登录时，从USTC获取GID
2. 检查GID是否在管理员列表中
3. 如果匹配，自动设置为管理员
4. 首次创建用户时，如果GID匹配管理员列表，创建为管理员

## 注意事项

1. **欢迎页面**：根路径 `/` 显示欢迎页面，不需要认证
2. **主应用**：访问 `/chat` 需要认证
3. **管理员页面**：访问 `/auth/admin` 需要管理员权限
4. **子路径支持**：所有路径都支持 `/nsrlchat` 前缀

## 测试流程

1. 访问 `https://nsrloa.ustc.edu.cn/nsrlchat/`
2. 看到欢迎页面
3. 点击"使用中科大统一身份认证登录"
4. 在USTC页面登录
5. 根据GID判断：
   - 如果是管理员GID → 跳转到管理员页面
   - 如果是普通用户GID → 跳转到主应用页面

