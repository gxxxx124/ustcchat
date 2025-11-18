# NSRL统一身份验证登录接入说明

## 概述

本系统使用 **CAS 2.0协议** 接入NSRL统一身份验证系统，实现单点登录（SSO）功能。

## CAS 2.0协议流程

### 1. 登录流程

```
用户 → 应用 → CAS服务器 → 用户输入账号密码 → CAS验证 → 重定向回应用（带ticket） → 应用验证ticket → 登录成功
```

### 2. 详细步骤

1. **用户点击登录按钮**
   - 前端调用 `/auth/cas/login`
   - 后端生成CAS登录URL，包含`service`参数（回调地址）
   - 重定向到CAS登录页面

2. **用户在CAS服务器登录**
   - 用户输入账号密码（如：`gx_20@mail.ustc.edu.cn`）
   - CAS服务器验证用户身份

3. **CAS服务器重定向回应用**
   - CAS验证成功后，重定向到`service`参数指定的回调地址
   - URL中包含`ticket`参数（一次性令牌）

4. **应用验证ticket**
   - 应用调用CAS的`serviceValidate`接口
   - 传入`service`和`ticket`参数
   - CAS返回XML格式的用户信息

5. **应用处理用户信息**
   - 解析XML，提取用户名、邮箱、GID等信息
   - 同步用户到本地数据库（如果不存在则创建）
   - 根据GID判断是否为管理员
   - 生成本地JWT token
   - 设置Cookie并重定向到主应用

## 代码实现

### 1. CAS客户端 (`nsrl_cas.py`)

```python
class NSRLCAS:
    """NSRL CAS 2.0客户端"""
    
    LOGIN_URL = "https://nsrloa.ustc.edu.cn/cas/login"
    SERVICE_VALIDATE_URL = "https://nsrloa.ustc.edu.cn/cas/serviceValidate"
    
    def get_login_url(self, state=None):
        """生成CAS登录URL"""
        params = {'service': self.service_url}
        login_url = f"{self.LOGIN_URL}?{urlencode(params)}"
        return login_url, state
    
    async def validate_ticket(self, ticket: str):
        """验证ticket并获取用户信息"""
        # 调用CAS的serviceValidate接口
        # 解析XML响应
        # 返回用户信息
```

### 2. 登录入口路由 (`auth_routes.py`)

```python
@auth_router.get("/cas/login")
async def cas_login(request: Request):
    """NSRL CAS登录入口"""
    # 1. 构建service_url（回调地址）
    base_url = os.getenv("BASE_URL", "")
    base_path = getBasePath(request)
    service_url = f"{base_url}{base_path}/auth/cas/callback"
    
    # 2. 更新CAS客户端的service_url
    nsrl_cas.service_url = service_url
    
    # 3. 生成CAS登录URL
    login_url, state = nsrl_cas.get_login_url()
    
    # 4. 重定向到CAS登录页面
    return RedirectResponse(url=login_url)
```

### 3. 回调处理路由 (`auth_routes.py`)

```python
@auth_router.get("/cas/callback")
async def cas_callback(request: Request, ticket: str, error: str = None):
    """NSRL CAS回调处理"""
    # 1. 验证ticket
    cas_data = await nsrl_cas.validate_ticket(ticket)
    
    # 2. 解析用户信息
    parsed_info = nsrl_cas.parse_user_info(cas_data)
    username = parsed_info.get('username', '')
    gid = parsed_info.get('gid', '')
    
    # 3. 判断是否为管理员（根据GID）
    is_admin = is_admin_gid(gid) if gid else False
    
    # 4. 同步用户到本地数据库
    user = await user_manager.get_user_by_username(username)
    if not user:
        # 创建新用户
        new_user = UserCreate(
            username=username,
            password=secrets.token_urlsafe(32),  # 随机密码
            email=parsed_info.get('email', username),
            is_admin=is_admin
        )
        user = await user_manager.create_user(new_user)
    
    # 5. 生成本地JWT token
    local_token = create_access_token(data={"sub": user.username})
    
    # 6. 设置Cookie并重定向
    response = RedirectResponse(url=redirect_url, status_code=303)
    response.set_cookie(
        key="access_token",
        value=local_token,
        path=cookie_path,
        max_age=1800,  # 30分钟
        httponly=False,
        samesite="lax",
        secure=is_https
    )
    
    return response
```

### 4. 应用初始化 (`web_memory.py`)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化
    base_url = os.getenv("BASE_URL", "https://nsrloa.ustc.edu.cn")
    init_nsrl_cas(base_url)
    # ...
```

## 关键配置

### 环境变量

```bash
# 基础URL（生产环境）
BASE_URL=https://nsrloa.ustc.edu.cn

# 子路径（如果部署在子路径下）
ROOT_PATH=/nsrlchat

# 管理员GID列表（用逗号分隔）
ADMIN_GIDS=gx_20,admin1,admin2
```

### CAS服务器端点

- **登录URL**: `https://nsrloa.ustc.edu.cn/cas/login`
- **Ticket验证**: `https://nsrloa.ustc.edu.cn/cas/serviceValidate`
- **登出URL**: `https://nsrloa.ustc.edu.cn/cas/logout`

### Service URL（回调地址）

格式：`{BASE_URL}{ROOT_PATH}/auth/cas/callback`

示例：
- 生产环境：`https://nsrloa.ustc.edu.cn/nsrlchat/auth/cas/callback`
- 本地开发：`http://localhost:8000/auth/cas/callback`

## 重要注意事项

### 1. Service URL必须完全一致

- 登录时使用的`service`参数
- 验证ticket时使用的`service`参数
- 必须**完全一致**（包括协议、域名、路径）

如果不一致，CAS会返回`INVALID_TICKET`错误。

### 2. Ticket是一次性的

- 每个ticket只能使用一次
- 验证后立即失效
- 不能重复使用

### 3. Cookie路径设置

由于应用部署在子路径`/nsrlchat/`下，需要设置正确的Cookie路径：

```python
# 主Cookie：在子路径下
response.set_cookie(
    key="access_token",
    value=local_token,
    path="/nsrlchat/",  # 子路径
    ...
)

# 备用Cookie：在根路径下（防止路径不匹配）
response.set_cookie(
    key="access_token_root",
    value=local_token,
    path="/",  # 根路径
    ...
)
```

### 4. 用户信息解析

CAS返回的用户信息格式：
- 用户名：通常是邮箱格式（如：`gx_20@mail.ustc.edu.cn`）
- GID：从用户名中提取（`@`前面的部分）
- 其他属性：从XML的`attributes`节点中提取

### 5. 管理员判断

根据GID判断是否为管理员：

```python
def is_admin_gid(gid: str) -> bool:
    """判断GID是否为管理员"""
    admin_gids = os.getenv("ADMIN_GIDS", "").split(",")
    return gid.strip() in [g.strip() for g in admin_gids if g.strip()]
```

## 前端集成

### 登录按钮

在登录页面添加CAS登录按钮：

```html
<button type="button" class="cas-login-button" onclick="casLogin()">
    <span>🔐</span>
    <span>使用NSRL统一身份认证登录</span>
</button>
```

### JavaScript函数

```javascript
function casLogin() {
    const basePath = getBasePath();  // 获取子路径
    const loginUrl = basePath + '/auth/cas/login';
    window.location.href = loginUrl;
}

function getBasePath() {
    const path = window.location.pathname;
    if (path.startsWith('/nsrlchat')) {
        return '/nsrlchat';
    }
    return '';
}
```

## 测试账号

- 用户名：`gx_20@mail.ustc.edu.cn`
- 密码：`Gx123456`

## 故障排查

### 1. Ticket验证失败

**错误**: `INVALID_TICKET` 或 `Ticket not recognized`

**原因**:
- Service URL不一致
- Ticket已过期或被使用过

**解决**:
- 检查登录时和验证时使用的`service`参数是否完全一致
- 确保使用环境变量`BASE_URL`和`ROOT_PATH`

### 2. 重定向到登录页面

**原因**:
- Cookie未设置成功
- Token验证失败
- 中间件拦截

**解决**:
- 检查Cookie路径是否正确
- 检查token是否有效（未过期、签名正确）
- 查看中间件日志

### 3. 无法跳转到CAS登录页面

**原因**:
- 路径错误
- 中间件拦截了`/auth/cas/login`

**解决**:
- 确保`/auth/cas/login`在中间件的排除列表中
- 检查`getBasePath()`函数是否正确

## 相关文件

- `nsrl_cas.py` - CAS客户端实现
- `auth_routes.py` - 认证路由（包含CAS登录和回调）
- `auth_middleware.py` - 认证中间件
- `auth.py` - Token生成和验证
- `web_memory.py` - 应用主文件（初始化CAS客户端）

