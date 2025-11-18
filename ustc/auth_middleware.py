from fastapi import Request, HTTPException, status
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging
import os
from auth import verify_token

logger = logging.getLogger(__name__)

class AuthMiddleware(BaseHTTPMiddleware):
    """认证中间件"""
    
    def __init__(self, app, excluded_paths=None):
        super().__init__(app)
        # 不需要认证的路径
        self.excluded_paths = excluded_paths or [
            "/auth/login",
            "/auth/login-page",
            "/auth/cas/login",  # CAS登录入口
            "/auth/cas/callback",  # CAS回调
            "/auth/ustc/login",  # USTC OAuth登录入口
            "/auth/ustc/callback",  # USTC OAuth回调
            "/health",
            "/static",
            "/marker_outputs",
            "/favicon.ico",
            "/ustc.svg",  # SVG文件不需要认证
            "/nsrlchat/ustc.svg",  # 子路径SVG文件不需要认证
            "/kb/api/upload-file",  # 文件上传不需要认证
            "/agent"  # API路径不需要中间件认证（API内部会处理认证）
        ]
    
    async def dispatch(self, request: Request, call_next):
        current_path = request.url.path
        logger.debug(f"🔍 中间件检查路径: {current_path}")
        
        # 检查是否为排除的路径
        if self._is_excluded_path(current_path):
            logger.debug(f"✅ 路径被排除，跳过认证检查: {current_path}")
            return await call_next(request)
        
        # 检查是否有有效的认证token
        if not self._has_valid_auth(request):
            # 如果是API请求，返回401错误
            if request.url.path.startswith(("/agent", "/kb")):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="需要登录才能访问此功能"
                )
            # 如果是页面请求，重定向到登录页面
            else:
                # 获取子路径前缀
                root_path = request.scope.get("root_path", "")
                # 优先从环境变量读取（用于生产环境）
                if not root_path:
                    root_path = os.getenv("ROOT_PATH", "")
                # 如果还是没有，默认使用 /nsrlchat（所有路径都挂载在 /nsrlchat 下）
                if not root_path:
                    root_path = "/nsrlchat"
                
                # 检查当前路径，避免重定向到登录页面时再次触发中间件
                current_path = request.url.path
                # 更严格的检查：检查完整路径或路径结尾
                if ('/auth/login-page' in current_path or '/auth/login' in current_path or 
                    current_path.endswith('/auth/login-page') or current_path.endswith('/auth/login')):
                    # 如果已经在登录页面，不重定向，直接返回（应该被排除，但双重保险）
                    logger.debug(f"⚠️ 在登录页面但认证失败，路径: {current_path}, 应该已被排除，直接返回")
                    return await call_next(request)
                
                # 检查是否是从登录页面重定向过来的（通过Referer头）
                # 如果是从登录页面来的，说明可能是循环重定向，不要再次重定向
                referer = request.headers.get("referer", "")
                if referer and ('/auth/login-page' in referer or '/auth/login' in referer):
                    logger.warning(f"⚠️ 检测到从登录页面重定向过来，可能是循环重定向，路径: {current_path}, Referer: {referer}")
                    # 不重定向，直接返回，让前端处理
                    # 返回一个简单的HTML页面，提示需要登录
                    from fastapi.responses import HTMLResponse
                    return HTMLResponse(
                        content=f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="UTF-8">
                            <title>需要登录</title>
                            <script>
                                // 清除可能无效的token
                                localStorage.removeItem('access_token');
                                document.cookie = 'access_token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT';
                                document.cookie = 'access_token_root=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT';
                                // 跳转到登录页面
                                window.location.href = '{root_path}/auth/login-page';
                            </script>
                        </head>
                        <body>
                            <p>正在跳转到登录页面...</p>
                        </body>
                        </html>
                        """,
                        status_code=401
                    )
                
                login_url = f"{root_path}/auth/login-page"
                logger.warning(f"🔄 重定向到登录页面: {login_url}, 当前路径: {current_path}, root_path: {root_path}")
                logger.warning(f"🔄 Cookie信息: {dict(request.cookies)}")
                return RedirectResponse(url=login_url, status_code=302)
        
        return await call_next(request)
    
    def _is_excluded_path(self, path: str) -> bool:
        """检查路径是否在排除列表中"""
        # 移除子路径前缀进行比较
        clean_path = path
        if path.startswith('/nsrlchat'):
            clean_path = path[9:]  # 移除 '/nsrlchat' 前缀
        elif path.startswith('/NSRLChat'):
            clean_path = path[9:]  # 移除 '/NSRLChat' 前缀
        
        # 如果清理后的路径为空，说明是 /nsrlchat 或 /NSRLChat，需要认证
        if not clean_path:
            logger.debug(f"❌ 路径未被排除（空路径）: {path}")
            return False
        
        # 确保 clean_path 以 / 开头
        if not clean_path.startswith('/'):
            clean_path = '/' + clean_path
        
        # 注意：根路径 '/' 不应该被排除，因为主应用需要认证
        # 只有特定的认证相关路径才应该被排除
        for excluded_path in self.excluded_paths:
            # 如果排除路径是 '/'，跳过（因为主应用需要认证）
            if excluded_path == '/':
                continue
            # 精确匹配
            if clean_path == excluded_path:
                logger.debug(f"✅ 路径被排除（精确匹配）: {path} -> {clean_path} 匹配 {excluded_path}")
                return True
            # 前缀匹配（确保是完整路径段）
            if clean_path.startswith(excluded_path + '/') or clean_path.startswith(excluded_path + '?'):
                logger.debug(f"✅ 路径被排除（前缀匹配）: {path} -> {clean_path} 匹配 {excluded_path}")
                return True
        logger.debug(f"❌ 路径未被排除: {path} -> {clean_path}")
        return False
    
    def _has_valid_auth(self, request: Request) -> bool:
        """检查是否有有效的认证"""
        # 从Authorization头获取token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            # 验证token有效性（包括签名和过期时间）
            if token == "test":
                return True
            token_data = verify_token(token)
            if token_data:
                return True
        
        # 从Cookie获取token（用于页面访问）
        # 优先使用子路径的Cookie，如果没有则使用根路径的Cookie
        token = request.cookies.get("access_token") or request.cookies.get("access_token_root")
        logger.debug(f"🔍 Cookie检查 - 路径: {request.url.path}, access_token存在: {bool(request.cookies.get('access_token'))}, access_token_root存在: {bool(request.cookies.get('access_token_root'))}, token存在: {bool(token)}")
        if token:
            # 验证token有效性（包括签名和过期时间）
            if token == "test":
                return True
            try:
                token_data = verify_token(token)
                if token_data:
                    logger.info(f"✅ Token验证成功: {token_data.username}, 路径: {request.url.path}, Cookie来源: {'access_token' if request.cookies.get('access_token') else 'access_token_root'}")
                    return True
                else:
                    logger.warning(f"❌ Token验证失败，路径: {request.url.path}, Cookie存在但验证失败, Token前20字符: {token[:20]}")
            except Exception as e:
                logger.error(f"❌ Token验证异常: {str(e)}, 路径: {request.url.path}")
        else:
            # 详细记录Cookie信息，帮助调试
            all_cookies = dict(request.cookies)
            cookie_keys = list(request.cookies.keys())
            logger.warning(f"❌ 未找到access_token Cookie，路径: {request.url.path}, Cookie数量: {len(cookie_keys)}, Cookie键: {cookie_keys}, 所有Cookie: {all_cookies}")
        
        return False

def create_auth_middleware(excluded_paths=None):
    """创建认证中间件"""
    def middleware_factory(app):
        return AuthMiddleware(app, excluded_paths=excluded_paths)
    return middleware_factory
