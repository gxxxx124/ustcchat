from fastapi import APIRouter, HTTPException, status, Depends, Request, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
import secrets
from auth import (
    UserCreate, UserLogin, UserResponse, Token, UserManager, UserRole,
    create_access_token, get_current_user, get_current_admin_user
)
from psycopg_pool import AsyncConnectionPool
from ustc_oauth import USTCOAuth
from nsrl_cas import NSRLCAS

logger = logging.getLogger(__name__)

# 创建认证路由
auth_router = APIRouter(prefix="/auth", tags=["认证"])

# 全局变量，将在主应用中设置
user_manager: UserManager = None
ustc_oauth: Optional[USTCOAuth] = None
nsrl_cas: Optional[NSRLCAS] = None

def set_user_manager(pool: AsyncConnectionPool):
    """设置用户管理器"""
    global user_manager
    user_manager = UserManager(pool)

def init_ustc_oauth(base_url: str = ""):
    """初始化USTC OAuth客户端"""
    global ustc_oauth
    
    # 从环境变量或配置文件读取
    client_id = os.getenv("USTC_CLIENT_ID", "")
    client_secret = os.getenv("USTC_CLIENT_SECRET", "")
    
    if not client_id or not client_secret:
        logger.warning("USTC OAuth配置未设置，USTC登录功能将不可用")
        return
    
    # 构建回调URL
    redirect_uri = f"{base_url}/auth/ustc/callback"
    ustc_oauth = USTCOAuth(client_id, client_secret, redirect_uri)
    logger.info(f"USTC OAuth初始化成功，回调地址: {redirect_uri}")

def init_nsrl_cas(base_url: str = ""):
    """初始化NSRL CAS客户端"""
    global nsrl_cas
    
    # 获取子路径（从环境变量）
    root_path = os.getenv("ROOT_PATH", "")
    
    # 构建回调URL
    if root_path:
        service_url = f"{base_url}{root_path}/auth/cas/callback"
    else:
        service_url = f"{base_url}/auth/cas/callback"
    
    nsrl_cas = NSRLCAS(service_url)
    logger.info(f"NSRL CAS初始化成功，回调地址: {service_url}")

# 管理员GID列表（从环境变量读取，用逗号分隔）
def get_admin_gids() -> List[str]:
    """获取管理员GID列表"""
    admin_gids_str = os.getenv("ADMIN_GIDS", "")
    if not admin_gids_str:
        return []
    return [gid.strip() for gid in admin_gids_str.split(",") if gid.strip()]

def is_admin_gid(gid: str) -> bool:
    """检查GID是否为管理员"""
    admin_gids = get_admin_gids()
    return gid in admin_gids

def getBasePath(request: Request) -> str:
    """从请求中获取基础路径"""
    # 优先从环境变量读取（用于生产环境）
    root_path = os.getenv("ROOT_PATH", "")
    if root_path:
        return root_path
    
    # 从请求路径推断
    path = request.url.path
    if path.startswith('/nsrlchat'):
        return '/nsrlchat'
    elif path.startswith('/NSRLChat'):
        return '/NSRLChat'
    
    # 默认返回 /nsrlchat（所有路径都挂载在 /nsrlchat 下）
    return '/nsrlchat'

@auth_router.get("/login")
async def login_redirect(request: Request):
    """重定向到登录页面"""
    base_path = getBasePath(request)
    return RedirectResponse(url=f"{base_path}/auth/login-page", status_code=302)

@auth_router.post("/login", response_model=Token)
async def login(user_login: UserLogin):
    """用户登录（API）"""
    if not user_manager:
        raise HTTPException(
            status_code=500,
            detail="用户管理器未初始化"
        )
    
    user = await user_manager.authenticate_user(
        user_login.username, user_login.password
    )
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@auth_router.get("/me", response_model=UserResponse)
async def get_current_user_info(request: Request, current_user: UserResponse = Depends(get_current_user)):
    """获取当前用户信息"""
    # 添加日志，帮助调试
    token_from_header = request.headers.get("Authorization", "")
    token_from_cookie = request.cookies.get("access_token") or request.cookies.get("access_token_root")
    logger.debug(f"/auth/me - Header token: {token_from_header[:30] if token_from_header else 'None'}..., Cookie token: {token_from_cookie[:30] if token_from_cookie else 'None'}...")
    # 添加用户角色日志
    logger.info(f"/auth/me - 用户: {current_user.username}, 角色: {current_user.role}, is_admin: {current_user.is_admin}")
    return current_user

@auth_router.post("/register", response_model=UserResponse)
async def register_user(user: UserCreate, current_user: UserResponse = Depends(get_current_admin_user)):
    """注册新用户（仅管理员）"""
    if not user_manager:
        raise HTTPException(
            status_code=500,
            detail="用户管理器未初始化"
        )
    
    try:
        new_user = await user_manager.create_user(user)
        logger.info(f"管理员 {current_user.username} 创建了新用户: {new_user.username}")
        return new_user
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"注册用户失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="注册用户失败"
        )

@auth_router.get("/users", response_model=List[UserResponse])
async def get_all_users(current_user: UserResponse = Depends(get_current_admin_user)):
    """获取所有用户列表（仅管理员）"""
    if not user_manager:
        raise HTTPException(
            status_code=500,
            detail="用户管理器未初始化"
        )
    
    try:
        users = await user_manager.get_all_users()
        return users
    except Exception as e:
        logger.error(f"获取用户列表失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="获取用户列表失败"
        )

@auth_router.delete("/users/{user_id}")
async def delete_user(user_id: int, current_user: UserResponse = Depends(get_current_admin_user)):
    """删除用户（仅管理员）"""
    if not user_manager:
        raise HTTPException(
            status_code=500,
            detail="用户管理器未初始化"
        )
    
    # 不能删除自己
    if user_id == current_user.id:
        raise HTTPException(
            status_code=400,
            detail="不能删除自己的账号"
        )
    
    try:
        success = await user_manager.delete_user(user_id)
        if success:
            logger.info(f"管理员 {current_user.username} 删除了用户ID: {user_id}")
            return {"message": "用户删除成功"}
        else:
            raise HTTPException(
                status_code=404,
                detail="用户不存在"
            )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"删除用户失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="删除用户失败"
        )

@auth_router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: int,
    role: str = Query(..., description="新角色: user, contributor, admin"),
    current_user: UserResponse = Depends(get_current_admin_user)
):
    """更新用户角色（仅管理员）"""
    if not user_manager:
        raise HTTPException(
            status_code=500,
            detail="用户管理器未初始化"
        )
    
    # 验证角色值
    valid_roles = [UserRole.USER, UserRole.CONTRIBUTOR, UserRole.ADMIN]
    if role not in valid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"无效的角色，必须是: {', '.join(valid_roles)}"
        )
    
    # 不能修改自己的角色
    if user_id == current_user.id:
        raise HTTPException(
            status_code=400,
            detail="不能修改自己的角色"
        )
    
    try:
        # 获取用户信息
        users = await user_manager.get_all_users()
        target_user = next((u for u in users if u.id == user_id), None)
        if not target_user:
            raise HTTPException(
                status_code=404,
                detail="用户不存在"
            )
        
        # 更新角色
        success = await user_manager.update_user_role(target_user.username, role)
        if success:
            logger.info(f"管理员 {current_user.username} 将用户 {target_user.username} 的角色更新为: {role}")
            return {"message": "用户角色更新成功", "role": role}
        else:
            raise HTTPException(
                status_code=500,
                detail="更新用户角色失败"
            )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"更新用户角色失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="更新用户角色失败"
        )

@auth_router.get("/ustc/login")
async def ustc_login(request: Request):
    """USTC统一身份认证登录入口"""
    if not ustc_oauth:
        raise HTTPException(
            status_code=500,
            detail="USTC OAuth未配置，请联系管理员"
        )
    
    # 获取基础URL
    base_url = str(request.base_url).rstrip('/')
    # 更新redirect_uri（如果base_url变化了）
    redirect_uri = f"{base_url}/auth/ustc/callback"
    ustc_oauth.redirect_uri = redirect_uri
    
    # 生成授权URL和state
    authorize_url, state = ustc_oauth.get_authorize_url()
    
    # 将state保存到session或通过其他方式保存（这里简化处理，实际应该使用session）
    # 为了简化，我们将state作为参数传递
    
    return RedirectResponse(url=authorize_url)

@auth_router.get("/ustc/callback")
async def ustc_callback(
    request: Request,
    code: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    error: Optional[str] = Query(None)
):
    """USTC OAuth回调处理"""
    if not ustc_oauth:
        raise HTTPException(
            status_code=500,
            detail="USTC OAuth未配置"
        )
    
    # 获取基础路径
    base_path = getBasePath(request)
    
    # 检查是否有错误
    if error:
        logger.error(f"USTC OAuth错误: {error}")
        return RedirectResponse(url=f"{base_path}/auth/login-page?error=ustc_auth_failed")
    
    if not code:
        return RedirectResponse(url=f"{base_path}/auth/login-page?error=no_code")
    
    try:
        # 1. 使用code获取access_token
        token_data = await ustc_oauth.get_access_token(code)
        access_token = token_data.get('access_token')
        
        if not access_token:
            logger.error("获取access_token失败")
            return RedirectResponse(url=f"{base_path}/auth/login-page?error=token_failed")
        
        # 2. 使用access_token获取用户信息
        user_data = await ustc_oauth.get_user_info(access_token)
        
        # 3. 解析用户信息
        parsed_info = ustc_oauth.parse_user_info(user_data)
        
        # 4. 检查是否为管理员（根据GID判断）
        gid = parsed_info.get('gid', '')
        is_admin = is_admin_gid(gid) if gid else False
        
        # 5. 同步用户到本地数据库
        username = parsed_info['username']
        if not username:
            logger.error("无法获取用户名")
            return RedirectResponse(url=f"{getBasePath(request)}/auth/login-page?error=no_username")
        
        # 检查用户是否存在，不存在则创建
        user = await user_manager.get_user_by_username(username)
        
        if not user:
            # 创建新用户，根据GID判断角色
            role = UserRole.ADMIN if is_admin else UserRole.USER
            new_user = UserCreate(
                username=username,
                password=secrets.token_urlsafe(32),  # 随机密码，因为使用USTC认证
                email=parsed_info.get('email'),
                is_admin=is_admin,
                role=role
            )
            try:
                user = await user_manager.create_user(new_user)
                logger.info(f"自动创建USTC用户: {username}, 角色: {role}")
            except HTTPException as e:
                if e.status_code == 400 and "已存在" in str(e.detail):
                    # 用户已存在，获取用户信息
                    user = await user_manager.get_user_by_username(username)
                    # 如果GID匹配管理员列表，更新用户为管理员
                    if is_admin and user.role != UserRole.ADMIN:
                        await user_manager.update_user_role(username, UserRole.ADMIN)
                        user = await user_manager.get_user_by_username(username)
                        logger.info(f"用户 {username} 的GID {gid} 匹配管理员列表，已更新为管理员")
                else:
                    raise
        else:
            # 用户已存在，检查是否需要更新角色
            if is_admin and user.role != UserRole.ADMIN:
                await user_manager.update_user_role(username, UserRole.ADMIN)
                user = await user_manager.get_user_by_username(username)
                logger.info(f"用户 {username} 的GID {gid} 匹配管理员列表，已更新为管理员")
        
        # 6. 创建本地JWT token
        local_token = create_access_token(data={"sub": user.username})
        
        # 7. 根据用户角色重定向到不同页面
        base_url = str(request.base_url).rstrip('/')
        base_path = getBasePath(request)
        
        # 判断跳转目标：管理员跳转到管理员页面，其他用户跳转到主应用
        if user.role == UserRole.ADMIN:
            redirect_url = f"{base_path}/auth/admin"
        else:
            redirect_url = f"{base_path}/"
        
        response = RedirectResponse(url=redirect_url)
        response.set_cookie(
            key="access_token",
            value=local_token,
            path="/",
            max_age=1800,  # 30分钟
            httponly=False,  # 允许前端JS读取
            samesite="lax"
        )
        
        logger.info(f"用户 {username} 登录成功，GID: {gid}, 角色: {user.role}, 跳转到: {redirect_url}")
        
        return response
        
    except Exception as e:
        logger.error(f"USTC OAuth回调处理失败: {str(e)}", exc_info=True)
        base_path = getBasePath(request)
        return RedirectResponse(url=f"{base_path}/auth/login-page?error=callback_failed")

@auth_router.get("/cas/login")
async def cas_login(request: Request):
    """NSRL CAS登录入口"""
    if not nsrl_cas:
        raise HTTPException(
            status_code=500,
            detail="NSRL CAS未配置"
        )
    
    # 获取基础URL和路径
    # 优先使用环境变量BASE_URL，确保与初始化时一致
    base_url = os.getenv("BASE_URL", "")
    if not base_url:
        base_url = f"{request.url.scheme}://{request.url.netloc}"
    
    base_path = getBasePath(request)
    
    # 构建service_url（必须与初始化时和验证时完全一致）
    if base_path:
        service_url = f"{base_url}{base_path}/auth/cas/callback"
    else:
        service_url = f"{base_url}/auth/cas/callback"
    
    # 更新service_url，确保与验证时一致
    nsrl_cas.service_url = service_url
    logger.debug(f"CAS登录 - 更新service_url: {service_url}")
    
    # 生成登录URL
    login_url, state = nsrl_cas.get_login_url()
    
    return RedirectResponse(url=login_url)

@auth_router.get("/cas/callback")
async def cas_callback(
    request: Request,
    ticket: Optional[str] = Query(None),
    error: Optional[str] = Query(None)
):
    """NSRL CAS回调处理"""
    if not nsrl_cas:
        raise HTTPException(
            status_code=500,
            detail="NSRL CAS未配置"
        )
    
    # 获取基础路径
    base_path = getBasePath(request)
    
    # 检查是否有错误
    if error:
        logger.error(f"NSRL CAS错误: {error}")
        return RedirectResponse(url=f"{base_path}/auth/login-page?error=cas_auth_failed")
    
    if not ticket:
        return RedirectResponse(url=f"{base_path}/auth/login-page?error=no_ticket")
    
    try:
        # 确保service_url与登录时一致（重要！）
        # 优先使用环境变量BASE_URL
        base_url = os.getenv("BASE_URL", "")
        if not base_url:
            base_url = f"{request.url.scheme}://{request.url.netloc}"
        
        if base_path:
            service_url = f"{base_url}{base_path}/auth/cas/callback"
        else:
            service_url = f"{base_url}/auth/cas/callback"
        
        # 更新service_url，确保验证时使用正确的URL
        nsrl_cas.service_url = service_url
        logger.debug(f"CAS回调 - 更新service_url: {service_url}, ticket: {ticket[:20]}...")
        
        # 1. 验证ticket并获取用户信息
        cas_data = await nsrl_cas.validate_ticket(ticket)
        
        if not cas_data.get('success'):
            error_msg = cas_data.get('error', '验证ticket失败')
            logger.error(f"CAS验证失败: {error_msg}")
            return RedirectResponse(url=f"{base_path}/auth/login-page?error=ticket_validation_failed")
        
        # 2. 解析用户信息
        parsed_info = nsrl_cas.parse_user_info(cas_data)
        
        # 3. 检查是否为管理员（根据GID判断）
        gid = parsed_info.get('gid', '')
        is_admin = is_admin_gid(gid) if gid else False
        
        # 4. 同步用户到本地数据库
        username = parsed_info.get('username', '')
        if not username:
            logger.error("无法获取用户名")
            return RedirectResponse(url=f"{base_path}/auth/login-page?error=no_username")
        
        # 检查用户是否存在，不存在则创建
        user = await user_manager.get_user_by_username(username)
        
        if not user:
            # 创建新用户，根据GID判断角色
            role = UserRole.ADMIN if is_admin else UserRole.USER
            new_user = UserCreate(
                username=username,
                password=secrets.token_urlsafe(32),  # 随机密码，因为使用CAS认证
                email=parsed_info.get('email', username),
                is_admin=is_admin,
                role=role
            )
            try:
                user = await user_manager.create_user(new_user)
                logger.info(f"自动创建CAS用户: {username}, 角色: {role}, GID: {gid}")
            except HTTPException as e:
                if e.status_code == 400 and "已存在" in str(e.detail):
                    # 用户已存在，获取用户信息
                    user = await user_manager.get_user_by_username(username)
                    # 如果GID匹配管理员列表，更新用户为管理员
                    if is_admin and user.role != UserRole.ADMIN:
                        await user_manager.update_user_role(username, UserRole.ADMIN)
                        user = await user_manager.get_user_by_username(username)
                        logger.info(f"用户 {username} 的GID {gid} 匹配管理员列表，已更新为管理员")
                else:
                    raise
        else:
            # 用户已存在，检查是否需要更新角色
            if is_admin and user.role != UserRole.ADMIN:
                await user_manager.update_user_role(username, UserRole.ADMIN)
                user = await user_manager.get_user_by_username(username)
                logger.info(f"用户 {username} 的GID {gid} 匹配管理员列表，已更新为管理员")
        
        # 5. 创建本地JWT token
        local_token = create_access_token(data={"sub": user.username})
        
        # 6. 根据用户角色重定向到不同页面
        # 判断跳转目标：管理员跳转到管理员页面，其他用户跳转到主应用
        # 注意：使用绝对路径，确保重定向正确
        # 优先使用环境变量BASE_URL，如果没有则从请求中获取
        base_url = os.getenv("BASE_URL", "")
        if not base_url:
            # 从请求中获取，但强制使用https（生产环境）
            scheme = "https" if "ustc.edu.cn" in request.url.netloc else request.url.scheme
            base_url = f"{scheme}://{request.url.netloc}"
        
        # 确保 base_path 是 /nsrlchat（所有路径都挂载在 /nsrlchat 下）
        if not base_path:
            base_path = "/nsrlchat"
        
        if user.role == UserRole.ADMIN:
            redirect_url = f"{base_url}{base_path}/auth/admin"
        else:
            redirect_url = f"{base_url}{base_path}/"
        
        response = RedirectResponse(url=redirect_url, status_code=303)  # 使用303 See Other，确保POST重定向为GET
        # 设置Cookie，确保在子路径下也能访问
        # 注意：Cookie的path应该设置为 /nsrlchat（不带末尾斜杠），这样访问 /nsrlchat/ 及其子路径时都会发送Cookie
        # 所有路径都挂载在 /nsrlchat 下，所以 Cookie path 必须是 /nsrlchat
        cookie_path = base_path  # 应该是 /nsrlchat
        # Cookie的path应该是 /nsrlchat（不带末尾斜杠）
        # 这样访问 /nsrlchat、/nsrlchat/ 和 /nsrlchat/xxx 时都会发送Cookie
        # 判断是否使用HTTPS
        is_https = base_url.startswith("https://")
        
        # 同时设置两个Cookie：一个在子路径，一个在根路径（作为备用）
        # 主Cookie：在子路径下（不带末尾斜杠，这样 /nsrlchat、/nsrlchat/ 和 /nsrlchat/xxx 都能匹配）
        response.set_cookie(
            key="access_token",
            value=local_token,
            path=cookie_path,  # 使用子路径作为Cookie路径（不带末尾斜杠）
            max_age=1800,  # 30分钟
            httponly=False,  # 允许前端JS读取
            samesite="lax",
            secure=is_https,  # HTTPS时设置为True
            domain=None  # 不设置domain，让浏览器自动使用当前域名
        )
        logger.info(f"设置主Cookie - access_token: path={cookie_path}, secure={is_https}, token前20字符: {local_token[:20]}")
        
        # 备用Cookie：在根路径下（防止nginx重定向导致路径不匹配）
        if base_path:
            response.set_cookie(
                key="access_token_root",
                value=local_token,
                path="/",  # 根路径
                max_age=1800,
                httponly=False,
                samesite="lax",
                secure=is_https,
                domain=None
            )
        
        # 添加详细的日志
        logger.info(f"设置Cookie - access_token: path={cookie_path}, secure={is_https}, domain=None")
        if base_path:
            logger.info(f"设置Cookie - access_token_root: path=/, secure={is_https}, domain=None")
        
        logger.info(f"用户 {username} CAS登录成功，GID: {gid}, 角色: {user.role}, 跳转到: {redirect_url}, Cookie路径: {cookie_path}, Token前20字符: {local_token[:20]}")
        
        return response
        
    except Exception as e:
        logger.error(f"NSRL CAS回调处理失败: {str(e)}", exc_info=True)
        base_path = getBasePath(request)
        return RedirectResponse(url=f"{base_path}/auth/login-page?error=cas_callback_failed")

@auth_router.get("/login-page", response_class=HTMLResponse)
async def login_page():
    """返回登录页面"""
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NSRLChat - 登录</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }

            .login-container {
                background: white;
                border-radius: 16px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                padding: 40px;
                width: 100%;
                max-width: 400px;
                position: relative;
                overflow: hidden;
            }

            .login-container::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(135deg, #00d4aa, #00a8cc);
            }

            .logo {
                text-align: center;
                margin-bottom: 30px;
            }

            .logo h1 {
                color: #333;
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 8px;
            }

            .logo p {
                color: #666;
                font-size: 14px;
            }

            .form-group {
                margin-bottom: 20px;
            }

            .form-group label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 500;
                font-size: 14px;
            }

            .form-group input {
                width: 100%;
                padding: 12px 16px;
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                font-size: 16px;
                transition: all 0.3s ease;
                background: #f8f9fa;
            }

            .form-group input:focus {
                outline: none;
                border-color: #00d4aa;
                background: white;
                box-shadow: 0 0 0 3px rgba(0, 212, 170, 0.1);
            }

            .login-button {
                width: 100%;
                padding: 14px;
                background: linear-gradient(135deg, #00d4aa, #00a8cc);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-bottom: 20px;
            }

            .login-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 212, 170, 0.3);
            }

            .login-button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }

            .error-message {
                background: #fee;
                color: #c33;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 20px;
                font-size: 14px;
                display: none;
            }

            .success-message {
                background: #efe;
                color: #363;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 20px;
                font-size: 14px;
                display: none;
            }

            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 2px solid #ffffff;
                border-radius: 50%;
                border-top-color: transparent;
                animation: spin 1s ease-in-out infinite;
                margin-right: 8px;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            .footer {
                text-align: center;
                color: #666;
                font-size: 12px;
                margin-top: 20px;
            }

            .admin-link {
                color: #00d4aa;
                text-decoration: none;
                font-weight: 500;
            }

            .admin-link:hover {
                text-decoration: underline;
            }

            /* NSRL CAS登录相关样式 */
            .cas-login-button {
                width: 100%;
                padding: 14px;
                background: #2563eb;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                margin-top: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
            }

            .cas-login-button:hover {
                background: #1d4ed8;
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(37, 99, 235, 0.3);
            }

            .divider {
                display: flex;
                align-items: center;
                margin: 20px 0;
                color: #999;
                font-size: 14px;
            }

            .divider::before,
            .divider::after {
                content: '';
                flex: 1;
                border-bottom: 1px solid #e1e5e9;
            }

            .divider span {
                padding: 0 12px;
            }
            */
        </style>
    </head>
    <body>
        <div class="login-container">
            <div class="logo">
                <h1>NSRLChat</h1>
                <p>智能对话助手</p>
            </div>

            <div id="errorMessage" class="error-message"></div>
            <div id="successMessage" class="success-message"></div>

            <form id="loginForm">
                <div class="form-group">
                    <label for="username">用户名</label>
                    <input type="text" id="username" name="username" required>
                </div>

                <div class="form-group">
                    <label for="password">密码</label>
                    <input type="password" id="password" name="password" required>
                </div>

                <button type="submit" class="login-button" id="loginButton">
                    <span id="buttonText">登录</span>
                </button>
            </form>

            <!-- NSRL CAS统一身份认证登录 -->
            <div class="divider">
                <span>或</span>
            </div>

            <button type="button" class="cas-login-button" onclick="casLogin()" id="casLoginButton">
                <span>🔐</span>
                <span>使用NSRL统一身份认证登录</span>
            </button>

            <div class="footer">
                <p>需要管理员权限？<a href="javascript:void(0)" onclick="window.location.href=getBasePath()+'/auth/admin'" class="admin-link">管理员登录</a></p>
            </div>
        </div>

        <script>
            // 获取基础路径，支持子路径部署
            function getBasePath() {
                const path = window.location.pathname;
                if (path.startsWith('/nsrlchat')) {
                    return '/nsrlchat';
                } else if (path.startsWith('/NSRLChat')) {
                    return '/NSRLChat';
                }
                return '';
            }
            
            const loginForm = document.getElementById('loginForm');
            const loginButton = document.getElementById('loginButton');
            const buttonText = document.getElementById('buttonText');
            const errorMessage = document.getElementById('errorMessage');
            const successMessage = document.getElementById('successMessage');

            function showError(message) {
                errorMessage.textContent = message;
                errorMessage.style.display = 'block';
                successMessage.style.display = 'none';
            }

            function showSuccess(message) {
                successMessage.textContent = message;
                successMessage.style.display = 'block';
                errorMessage.style.display = 'none';
            }

            function hideMessages() {
                errorMessage.style.display = 'none';
                successMessage.style.display = 'none';
            }

            function setLoading(loading) {
                if (loading) {
                    loginButton.disabled = true;
                    buttonText.innerHTML = '<span class="loading"></span>登录中...';
                } else {
                    loginButton.disabled = false;
                    buttonText.textContent = '登录';
                }
            }

            loginForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                hideMessages();

                const username = document.getElementById('username').value.trim();
                const password = document.getElementById('password').value;

                if (!username || !password) {
                    showError('请输入用户名和密码');
                    return;
                }

                setLoading(true);

                try {
                    const response = await fetch(getBasePath() + '/auth/login', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            username: username,
                            password: password
                        })
                    });

                    const data = await response.json();

                    if (response.ok) {
                        // 保存token到localStorage和Cookie
                        localStorage.setItem('access_token', data.access_token);
                        document.cookie = `access_token=${data.access_token}; path=/; max-age=1800`; // 30分钟过期
                        showSuccess('登录成功，正在跳转...');
                        
                        // 检查用户权限并跳转
                        setTimeout(async () => {
                            try {
                                // 获取用户信息
                                const userResponse = await fetch(getBasePath() + '/auth/me', {
                                    headers: {
                                        'Authorization': `Bearer ${data.access_token}`
                                    }
                                });
                                
                                if (userResponse.ok) {
                                    const user = await userResponse.json();
                                    // 如果是管理员，跳转到管理员页面；否则跳转到主页面
                                    if (user.role === 'admin') {
                                        window.location.href = getBasePath() + '/auth/admin';
                                    } else {
                                        window.location.href = getBasePath() + '/';
                                    }
                                } else {
                                    // 如果获取用户信息失败，直接跳转到主页面
                                    window.location.href = getBasePath() + '/';
                                }
                            } catch (error) {
                                console.error('获取用户信息失败:', error);
                                // 出错时跳转到主页面
                                window.location.href = getBasePath() + '/';
                            }
                        }, 1000);
                    } else {
                        showError(data.detail || '登录失败');
                    }
                } catch (error) {
                    console.error('登录错误:', error);
                    showError('网络错误，请稍后重试');
                } finally {
                    setLoading(false);
                }
            });

            // 检查是否已经登录（只在有token时检查，避免循环）
            window.addEventListener('load', () => {
                console.log('登录页面 load 事件触发，当前路径:', window.location.pathname);
                
                // 获取token的辅助函数（优先从Cookie获取）
                function getAccessToken() {
                    const cookies = document.cookie.split(';');
                    for (let cookie of cookies) {
                        const trimmed = cookie.trim();
                        const equalIndex = trimmed.indexOf('=');
                        if (equalIndex === -1) continue;
                        
                        const name = trimmed.substring(0, equalIndex).trim();
                        const value = trimmed.substring(equalIndex + 1).trim();
                        
                        if (name === 'access_token' || name === 'access_token_root') {
                            let tokenValue = value;
                            if ((tokenValue.startsWith('"') && tokenValue.endsWith('"')) || 
                                (tokenValue.startsWith("'") && tokenValue.endsWith("'"))) {
                                tokenValue = tokenValue.slice(1, -1);
                            }
                            localStorage.setItem('access_token', tokenValue);
                            console.log('登录页面 - 从Cookie获取token');
                            return tokenValue;
                        }
                    }
                    const storedToken = localStorage.getItem('access_token');
                    if (storedToken) {
                        console.log('登录页面 - 从localStorage获取token');
                    } else {
                        console.log('登录页面 - 没有token，保持在登录页面');
                    }
                    return storedToken;
                }
                
                const token = getAccessToken();
                // 只有在有token时才验证，避免无token时触发重定向循环
                // 但是，如果URL中有error参数，说明是认证失败后的重定向，不要自动跳转
                const urlParams = new URLSearchParams(window.location.search);
                const hasError = urlParams.has('error');
                
                if (token && !hasError) {
                    console.log('登录页面 - 检测到token，验证中...');
                    // 验证token是否有效
                    fetch(getBasePath() + '/auth/me', {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    })
                    .then(response => {
                        console.log('登录页面 - /auth/me 响应状态:', response.status);
                        if (response.ok) {
                            return response.json().then(user => {
                                // 已登录，根据角色跳转
                                console.log('登录页面 - token有效，用户角色:', user.role);
                            // 使用 replace 而不是 href，避免在历史记录中留下记录
                                if (user.role === 'admin') {
                                    window.location.replace(getBasePath() + '/auth/admin');
                                } else {
                            window.location.replace(getBasePath() + '/');
                                }
                            });
                        } else {
                            // token无效，清除（但不跳转，保持在登录页面）
                            console.log('登录页面 - token无效，清除token，保持在登录页面');
                            localStorage.removeItem('access_token');
                            // 清除Cookie
                            document.cookie = 'access_token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT';
                            document.cookie = 'access_token_root=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT';
                        }
                    })
                    .catch((error) => {
                        // 出错时清除token，但不跳转
                        console.error('登录页面 - /auth/me 请求失败:', error);
                        localStorage.removeItem('access_token');
                        document.cookie = 'access_token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT';
                        document.cookie = 'access_token_root=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT';
                    });
                } else {
                    if (hasError) {
                        console.log('登录页面 - URL中有error参数，不自动跳转');
                } else {
                    console.log('登录页面 - 没有token，不做任何操作，保持在登录页面');
                    }
                }
            });

            // NSRL CAS统一身份认证登录
            function casLogin() {
                try {
                    const basePath = getBasePath();
                    const loginUrl = basePath + '/auth/cas/login';
                    console.log('CAS登录URL:', loginUrl);
                    window.location.href = loginUrl;
                } catch (error) {
                    console.error('CAS登录错误:', error);
                    alert('跳转失败，请检查控制台错误信息');
                }
            }

            // 检查URL参数中的错误信息（CAS相关）
            window.addEventListener('load', () => {
                const urlParams = new URLSearchParams(window.location.search);
                const error = urlParams.get('error');
                
                if (error) {
                    let errorMessage = '登录失败';
                    switch(error) {
                        case 'cas_auth_failed':
                            errorMessage = 'NSRL统一身份认证失败';
                            break;
                        case 'no_ticket':
                            errorMessage = '未获取到ticket';
                            break;
                        case 'ticket_validation_failed':
                            errorMessage = 'ticket验证失败';
                            break;
                        case 'no_username':
                            errorMessage = '无法获取用户信息';
                            break;
                        case 'cas_callback_failed':
                            errorMessage = '回调处理失败，请重试';
                            break;
                    }
                    showError(errorMessage);
                    
                    // 清除URL中的错误参数
                    window.history.replaceState({}, document.title, window.location.pathname);
                }
            });
        </script>
    </body>
    </html>
    """

@auth_router.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """返回管理员页面"""
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NSRLChat - 管理员</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
                background: #f5f7fa;
                min-height: 100vh;
                padding: 20px;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
            }

            .header {
                background: white;
                border-radius: 12px;
                padding: 24px;
                margin-bottom: 24px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .header h1 {
                color: #333;
                font-size: 24px;
                font-weight: 700;
            }

            .user-info {
                display: flex;
                align-items: center;
                gap: 16px;
            }

            .user-name {
                color: #666;
                font-size: 14px;
            }

            .chat-btn {
                background: linear-gradient(135deg, #00d4aa, #00a8cc);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.2s ease;
                margin-right: 12px;
            }

            .chat-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 212, 170, 0.3);
            }

            .file-btn {
                background: linear-gradient(135deg, #8b5cf6, #a855f7);
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.2s ease;
                margin-right: 12px;
            }

            .file-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
            }

            .logout-btn {
                background: #ff4757;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.2s ease;
            }

            .logout-btn:hover {
                background: #ff3742;
            }

            .main-content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
            }

            .card {
                background: white;
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }

            .card h2 {
                color: #333;
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 20px;
            }

            .form-group {
                margin-bottom: 16px;
            }

            .form-group label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 500;
                font-size: 14px;
            }

            .form-group input {
                width: 100%;
                padding: 10px 12px;
                border: 2px solid #e1e5e9;
                border-radius: 6px;
                font-size: 14px;
                transition: all 0.2s ease;
            }

            .form-group input:focus {
                outline: none;
                border-color: #00d4aa;
            }

            .checkbox-group {
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 16px;
            }

            .checkbox-group input[type="checkbox"] {
                width: auto;
            }

            .btn {
                background: linear-gradient(135deg, #00d4aa, #00a8cc);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.2s ease;
            }

            .btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 212, 170, 0.3);
            }

            .btn:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }

            .users-list {
                max-height: 400px;
                overflow-y: auto;
            }

            .user-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px;
                border: 1px solid #e1e5e9;
                border-radius: 6px;
                margin-bottom: 8px;
                background: #f8f9fa;
            }

            .user-info-item {
                flex: 1;
            }

            .user-name {
                font-weight: 500;
                color: #333;
                margin-bottom: 4px;
            }

            .user-details {
                font-size: 12px;
                color: #666;
            }

            .admin-badge {
                background: #00d4aa;
                color: white;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 10px;
                font-weight: 500;
                margin-left: 8px;
            }

            .contributor-badge {
                background: #8b5cf6;
                color: white;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 10px;
                font-weight: 500;
                margin-left: 8px;
            }

            .delete-btn {
                background: #ff4757;
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.2s ease;
            }

            .delete-btn:hover {
                background: #ff3742;
            }

            .message {
                padding: 12px;
                border-radius: 6px;
                margin-bottom: 16px;
                font-size: 14px;
                display: none;
            }

            .message.error {
                background: #fee;
                color: #c33;
                border: 1px solid #fcc;
            }

            .message.success {
                background: #efe;
                color: #363;
                border: 1px solid #cfc;
            }

            .loading {
                display: inline-block;
                width: 16px;
                height: 16px;
                border: 2px solid #ffffff;
                border-radius: 50%;
                border-top-color: transparent;
                animation: spin 1s ease-in-out infinite;
                margin-right: 8px;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            @media (max-width: 768px) {
                .main-content {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>NSRLChat 管理员控制台</h1>
                <div class="user-info">
                    <button class="chat-btn" onclick="goToChat()">💬 进入对话</button>
                    <button class="file-btn" onclick="goToFileManager()">📁 文件管理</button>
                    <span class="user-name" id="currentUser">加载中...</span>
                    <button class="logout-btn" onclick="logout()">退出登录</button>
                </div>
            </div>

            <div class="main-content">
                <div class="card">
                    <h2>添加新用户</h2>
                    <div id="message" class="message"></div>
                    
                    <form id="addUserForm">
                        <div class="form-group">
                            <label for="username">用户名</label>
                            <input type="text" id="username" name="username" required>
                        </div>

                        <div class="form-group">
                            <label for="password">密码</label>
                            <input type="password" id="password" name="password" required>
                        </div>

                        <div class="form-group">
                            <label for="email">邮箱（可选）</label>
                            <input type="email" id="email" name="email">
                        </div>

                        <div class="form-group">
                            <label for="userRole">用户角色</label>
                            <select id="userRole" name="userRole" class="form-select">
                                <option value="user">普通用户</option>
                                <option value="contributor">知识库贡献者</option>
                                <option value="admin">管理员</option>
                            </select>
                        </div>

                        <button type="submit" class="btn" id="addUserBtn">
                            <span id="addUserBtnText">添加用户</span>
                        </button>
                    </form>
                </div>

                <div class="card">
                    <h2>用户列表</h2>
                    <div class="users-list" id="usersList">
                        <div style="text-align: center; color: #666; padding: 20px;">
                            加载中...
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // 获取基础路径，支持子路径部署
            function getBasePath() {
                const path = window.location.pathname;
                if (path.startsWith('/nsrlchat')) {
                    return '/nsrlchat';
                } else if (path.startsWith('/NSRLChat')) {
                    return '/NSRLChat';
                }
                return '';
            }
            
            let currentUser = null;

            function showMessage(message, type) {
                const messageEl = document.getElementById('message');
                messageEl.textContent = message;
                messageEl.className = `message ${type}`;
                messageEl.style.display = 'block';
                
                setTimeout(() => {
                    messageEl.style.display = 'none';
                }, 5000);
            }

            function setLoading(loading) {
                const btn = document.getElementById('addUserBtn');
                const btnText = document.getElementById('addUserBtnText');
                
                if (loading) {
                    btn.disabled = true;
                    btnText.innerHTML = '<span class="loading"></span>添加中...';
                } else {
                    btn.disabled = false;
                    btnText.textContent = '添加用户';
                }
            }

            // 获取token的辅助函数（优先从Cookie获取，如果没有则从localStorage获取）
            function getAccessToken() {
                // 首先尝试从Cookie获取（CAS登录使用Cookie）
                const cookies = document.cookie.split(';');
                for (let cookie of cookies) {
                    const trimmed = cookie.trim();
                    const equalIndex = trimmed.indexOf('=');
                    if (equalIndex === -1) continue;
                    
                    const name = trimmed.substring(0, equalIndex).trim();
                    const value = trimmed.substring(equalIndex + 1).trim();
                    
                    if (name === 'access_token' || name === 'access_token_root') {
                        // 移除可能的引号
                        let tokenValue = value;
                        if ((tokenValue.startsWith('"') && tokenValue.endsWith('"')) || 
                            (tokenValue.startsWith("'") && tokenValue.endsWith("'"))) {
                            tokenValue = tokenValue.slice(1, -1);
                        }
                        // 同时保存到localStorage，方便后续使用
                        localStorage.setItem('access_token', tokenValue);
                        console.log('从Cookie获取token:', name, tokenValue.substring(0, 30) + '...');
                        return tokenValue;
                    }
                }
                // 如果Cookie中没有，尝试从localStorage获取（传统登录使用）
                const storedToken = localStorage.getItem('access_token');
                if (storedToken) {
                    console.log('从localStorage获取token:', storedToken.substring(0, 30) + '...');
                }
                return storedToken;
            }

            async function loadCurrentUser() {
                try {
                    const token = getAccessToken();
                    if (!token) {
                        window.location.href = getBasePath() + '/auth/login-page';
                        return;
                    }

                    const response = await fetch(getBasePath() + '/auth/me', {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    });

                    if (!response.ok) {
                        localStorage.removeItem('access_token');
                        window.location.href = getBasePath() + '/auth/login-page';
                        return;
                    }

                    currentUser = await response.json();
                    document.getElementById('currentUser').textContent = currentUser.username;
                } catch (error) {
                    console.error('加载用户信息失败:', error);
                    window.location.href = getBasePath() + '/auth/login-page';
                }
            }

            async function loadUsers() {
                try {
                    const token = getAccessToken();
                    if (!token) {
                        throw new Error('未找到token');
                    }
                    const response = await fetch(getBasePath() + '/auth/users', {
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    });

                    if (!response.ok) {
                        throw new Error('加载用户列表失败');
                    }

                    const users = await response.json();
                    const usersList = document.getElementById('usersList');
                    
                    if (users.length === 0) {
                        usersList.innerHTML = '<div style="text-align: center; color: #666; padding: 20px;">暂无用户</div>';
                        return;
                    }

                    usersList.innerHTML = users.map(user => `
                        <div class="user-item">
                            <div class="user-info-item">
                                <div class="user-name">
                                    ${user.username}
                                    ${user.role === 'admin' ? '<span class="admin-badge">管理员</span>' : 
                                      user.role === 'contributor' ? '<span class="contributor-badge">知识库贡献者</span>' : ''}
                                </div>
                                <div class="user-details">
                                    创建时间: ${new Date(user.created_at).toLocaleString()}
                                    ${user.last_login ? `<br>最后登录: ${new Date(user.last_login).toLocaleString()}` : ''}
                                </div>
                            </div>
                            ${user.id !== currentUser.id ? 
                                `<div style="display: flex; gap: 8px;">
                                    <select class="role-select" onchange="updateUserRole(${user.id}, this.value)" style="padding: 4px 8px; border-radius: 4px; border: 1px solid #d1d5db; font-size: 12px;">
                                        <option value="user" ${user.role === 'user' ? 'selected' : ''}>普通用户</option>
                                        <option value="contributor" ${user.role === 'contributor' ? 'selected' : ''}>知识库贡献者</option>
                                        <option value="admin" ${user.role === 'admin' ? 'selected' : ''}>管理员</option>
                                    </select>
                                    <button class="delete-btn" onclick="deleteUser(${user.id}, '${user.username}')">删除</button>
                                </div>` : 
                                '<span style="color: #999; font-size: 12px;">当前用户</span>'
                            }
                        </div>
                    `).join('');
                } catch (error) {
                    console.error('加载用户列表失败:', error);
                    document.getElementById('usersList').innerHTML = 
                        '<div style="text-align: center; color: #c33; padding: 20px;">加载失败</div>';
                }
            }

            async function addUser(event) {
                event.preventDefault();
                
                const formData = new FormData(event.target);
                const userData = {
                    username: formData.get('username'),
                    password: formData.get('password'),
                    email: formData.get('email') || null,
                    role: formData.get('userRole') || 'user',
                    is_admin: formData.get('userRole') === 'admin'
                };

                if (!userData.username || !userData.password) {
                    showMessage('请填写用户名和密码', 'error');
                    return;
                }

                setLoading(true);

                try {
                    const token = getAccessToken();
                    if (!token) {
                        throw new Error('未找到token');
                    }
                    const response = await fetch(getBasePath() + '/auth/register', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${token}`
                        },
                        body: JSON.stringify(userData)
                    });

                    const data = await response.json();

                    if (response.ok) {
                        showMessage('用户添加成功', 'success');
                        event.target.reset();
                        loadUsers();
                    } else {
                        showMessage(data.detail || '添加用户失败', 'error');
                    }
                } catch (error) {
                    console.error('添加用户失败:', error);
                    showMessage('网络错误，请稍后重试', 'error');
                } finally {
                    setLoading(false);
                }
            }

            async function deleteUser(userId, username) {
                if (!confirm(`确定要删除用户 "${username}" 吗？此操作不可撤销。`)) {
                    return;
                }
                
                try {
                    const token = getAccessToken();
                    if (!token) {
                        throw new Error('未找到token');
                    }
                    const response = await fetch(`${getBasePath()}/auth/users/${userId}`, {
                        method: 'DELETE',
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    });

                    const data = await response.json();

                    if (response.ok) {
                        showMessage('用户删除成功', 'success');
                        loadUsers();
                    } else {
                        showMessage(data.detail || '删除用户失败', 'error');
                    }
                } catch (error) {
                    console.error('删除用户失败:', error);
                    showMessage('网络错误，请稍后重试', 'error');
                }
            }

            async function updateUserRole(userId, newRole) {
                if (!confirm(`确定要将该用户的角色更改为"${newRole === 'user' ? '普通用户' : newRole === 'contributor' ? '知识库贡献者' : '管理员'}"吗？`)) {
                    // 如果取消，重新加载用户列表以恢复原来的选择
                    loadUsers();
                    return;
                }
                
                try {
                    const token = getAccessToken();
                    if (!token) {
                        throw new Error('未找到token');
                    }
                    const response = await fetch(`${getBasePath()}/auth/users/${userId}/role?role=${newRole}`, {
                        method: 'PUT',
                        headers: {
                            'Authorization': `Bearer ${token}`
                        }
                    });

                    const data = await response.json();

                    if (response.ok) {
                        showMessage('用户角色更新成功', 'success');
                        loadUsers();
                    } else {
                        showMessage(data.detail || '更新用户角色失败', 'error');
                        loadUsers(); // 重新加载以恢复原来的选择
                    }
                } catch (error) {
                    console.error('更新用户角色失败:', error);
                    showMessage('网络错误，请稍后重试', 'error');
                    loadUsers(); // 重新加载以恢复原来的选择
                }
            }

            function goToChat() {
                // 跳转到主对话页面
                window.location.href = getBasePath() + '/';
            }

            function goToFileManager() {
                // 跳转到文件管理页面
                window.location.href = getBasePath() + '/upload.html';
            }

            function logout() {
                // 清除localStorage
                localStorage.removeItem('access_token');
                
                // 获取基础路径
                const basePath = getBasePath();
                
                // 清除所有Cookie（包括access_token和access_token_root）
                const cookiePaths = ['/', basePath ? basePath : '/'];
                const cookieNames = ['access_token', 'access_token_root'];
                
                for (const cookieName of cookieNames) {
                    for (const cookiePath of cookiePaths) {
                        // 清除Cookie（设置过期时间为过去）
                        document.cookie = `${cookieName}=; path=${cookiePath}; expires=Thu, 01 Jan 1970 00:00:00 GMT`;
                        // 也尝试清除带斜杠的路径
                        if (cookiePath !== '/') {
                            document.cookie = `${cookieName}=; path=${cookiePath}/; expires=Thu, 01 Jan 1970 00:00:00 GMT`;
                        }
                    }
                }
                
                console.log('已清除所有Cookie和localStorage');
                
                // 重定向到登录页面
                window.location.href = getBasePath() + '/auth/login-page';
            }

            // 页面加载时初始化
            document.addEventListener('DOMContentLoaded', async () => {
                await loadCurrentUser();
                await loadUsers();
                
                document.getElementById('addUserForm').addEventListener('submit', addUser);
            });
        </script>
    </body>
    </html>
    """
