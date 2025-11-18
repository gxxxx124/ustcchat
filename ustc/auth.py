from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any
import hashlib
import secrets
import datetime
import base64
import json
from psycopg_pool import AsyncConnectionPool
import logging
import os

logger = logging.getLogger(__name__)

# JWT配置（从环境变量读取）
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# 安全配置
security = HTTPBearer()

# 用户角色枚举
class UserRole:
    USER = "user"  # 普通用户：只能进入对话
    CONTRIBUTOR = "contributor"  # 知识库贡献者：能上传文档和进行对话
    ADMIN = "admin"  # 管理员：能赋予人知识库贡献者身份

# 数据模型
class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    is_admin: bool = False  # 向后兼容，将被role替代
    role: str = UserRole.USER  # 新角色字段

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    is_admin: bool  # 向后兼容，从role计算得出
    role: str  # 新角色字段
    created_at: str
    last_login: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# 密码加密
def hash_password(password: str) -> str:
    """使用SHA-256加密密码"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return hash_password(plain_password) == hashed_password

# 简单的令牌操作（替代JWT）
def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    """创建访问令牌"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire.isoformat()})
    
    # 创建简单的令牌：base64编码的JSON + 签名
    payload = json.dumps(to_encode).encode('utf-8')
    payload_b64 = base64.b64encode(payload).decode('utf-8')
    
    # 创建签名
    signature = hashlib.sha256((payload_b64 + SECRET_KEY).encode('utf-8')).hexdigest()
    
    # 组合令牌
    token = f"{payload_b64}.{signature}"
    return token

def verify_token(token: str) -> Optional[TokenData]:
    """验证令牌"""
    try:
        # 分割令牌
        parts = token.split('.')
        if len(parts) != 2:
            return None
        
        payload_b64, signature = parts
        
        # 验证签名
        expected_signature = hashlib.sha256((payload_b64 + SECRET_KEY).encode('utf-8')).hexdigest()
        if signature != expected_signature:
            return None
        
        # 解码载荷
        payload_bytes = base64.b64decode(payload_b64.encode('utf-8'))
        payload = json.loads(payload_bytes.decode('utf-8'))
        
        # 检查过期时间
        exp_str = payload.get("exp")
        if exp_str:
            exp_time = datetime.datetime.fromisoformat(exp_str)
            if datetime.datetime.utcnow() > exp_time:
                return None
        
        username: str = payload.get("sub")
        if username is None:
            return None
        token_data = TokenData(username=username)
        return token_data
    except Exception:
        return None

# 数据库操作
class UserManager:
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def create_user(self, user: UserCreate) -> UserResponse:
        """创建用户"""
        hashed_password = hash_password(user.password)
        
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    # 检查用户名是否已存在
                    await cur.execute(
                        "SELECT id FROM users WHERE username = %s",
                        (user.username,)
                    )
                    if await cur.fetchone():
                        raise HTTPException(
                            status_code=400,
                            detail="用户名已存在"
                        )
                    
                    # 确定角色：如果is_admin为True，则设为admin；否则使用role字段
                    if user.is_admin:
                        role = UserRole.ADMIN
                    else:
                        role = user.role if user.role else UserRole.USER
                    
                    # 创建用户
                    await cur.execute(
                        """
                        INSERT INTO users (username, password_hash, email, is_admin, role, created_at)
                        VALUES (%s, %s, %s, %s, %s, NOW())
                        RETURNING id, username, email, is_admin, role, created_at
                        """,
                        (user.username, hashed_password, user.email, user.is_admin, role)
                    )
                    
                    result = await cur.fetchone()
                    if not result:
                        raise HTTPException(
                            status_code=500,
                            detail="创建用户失败"
                        )
                    
                    return UserResponse(
                        id=result[0],
                        username=result[1],
                        email=result[2],
                        is_admin=result[3],
                        role=result[4] if result[4] else (UserRole.ADMIN if result[3] else UserRole.USER),
                        created_at=result[5].isoformat(),
                        last_login=None
                    )
                    
                except HTTPException:
                    raise
                except Exception as e:
                    logger.error(f"创建用户失败: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail="创建用户失败"
                    )

    async def authenticate_user(self, username: str, password: str) -> Optional[UserResponse]:
        """验证用户登录"""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    await cur.execute(
                        """
                        SELECT id, username, password_hash, email, is_admin, role, created_at, last_login
                        FROM users WHERE username = %s
                        """,
                        (username,)
                    )
                    
                    result = await cur.fetchone()
                    if not result:
                        return None
                    
                    user_id, username, password_hash, email, is_admin, role, created_at, last_login = result
                    
                    if not verify_password(password, password_hash):
                        return None
                    
                    # 更新最后登录时间
                    await cur.execute(
                        "UPDATE users SET last_login = NOW() WHERE id = %s",
                        (user_id,)
                    )
                    
                    # 如果没有role字段，从is_admin推断
                    if not role:
                        role = UserRole.ADMIN if is_admin else UserRole.USER
                    
                    return UserResponse(
                        id=user_id,
                        username=username,
                        email=email,
                        is_admin=is_admin,
                        role=role,
                        created_at=created_at.isoformat(),
                        last_login=datetime.datetime.now().isoformat()
                    )
                    
                except Exception as e:
                    logger.error(f"用户认证失败: {str(e)}")
                    return None

    async def get_user_by_username(self, username: str) -> Optional[UserResponse]:
        """根据用户名获取用户信息"""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    await cur.execute(
                        """
                        SELECT id, username, email, is_admin, role, created_at, last_login
                        FROM users WHERE username = %s
                        """,
                        (username,)
                    )
                    
                    result = await cur.fetchone()
                    if not result:
                        return None
                    
                    # 如果没有role字段，从is_admin推断
                    role = result[4] if result[4] else (UserRole.ADMIN if result[3] else UserRole.USER)
                    
                    return UserResponse(
                        id=result[0],
                        username=result[1],
                        email=result[2],
                        is_admin=result[3],
                        role=role,
                        created_at=result[5].isoformat(),
                        last_login=result[6].isoformat() if result[6] else None
                    )
                    
                except Exception as e:
                    logger.error(f"获取用户信息失败: {str(e)}")
                    return None

    async def get_all_users(self) -> list[UserResponse]:
        """获取所有用户列表（仅管理员）"""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    await cur.execute(
                        """
                        SELECT id, username, email, is_admin, role, created_at, last_login
                        FROM users ORDER BY created_at DESC
                        """
                    )
                    
                    results = await cur.fetchall()
                    users = []
                    
                    for result in results:
                        # 如果没有role字段，从is_admin推断
                        role = result[4] if result[4] else (UserRole.ADMIN if result[3] else UserRole.USER)
                        users.append(UserResponse(
                            id=result[0],
                            username=result[1],
                            email=result[2],
                            is_admin=result[3],
                            role=role,
                            created_at=result[5].isoformat(),
                            last_login=result[6].isoformat() if result[6] else None
                        ))
                    
                    return users
                    
                except Exception as e:
                    logger.error(f"获取用户列表失败: {str(e)}")
                    return []

    async def delete_user(self, user_id: int) -> bool:
        """删除用户（仅管理员）"""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    await cur.execute(
                        "DELETE FROM users WHERE id = %s",
                        (user_id,)
                    )
                    return cur.rowcount > 0
                    
                except Exception as e:
                    logger.error(f"删除用户失败: {str(e)}")
                    return False

    async def update_user_role(self, username: str, role: str) -> bool:
        """更新用户角色"""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                try:
                    # 更新角色，同时更新is_admin字段以保持向后兼容
                    is_admin = (role == UserRole.ADMIN)
                    await cur.execute(
                        """
                        UPDATE users 
                        SET role = %s, is_admin = %s 
                        WHERE username = %s
                        """,
                        (role, is_admin, username)
                    )
                    await conn.commit()
                    return cur.rowcount > 0
                    
                except Exception as e:
                    logger.error(f"更新用户角色失败: {str(e)}")
                    return False

# 全局变量存储数据库连接池
global_pool: Optional[AsyncConnectionPool] = None

def set_global_pool(pool: AsyncConnectionPool):
    """设置全局数据库连接池"""
    global global_pool
    global_pool = pool

# 依赖注入
async def get_token_from_request(request: Request) -> str:
    """从请求中获取token（支持Authorization头和Cookie）"""
    # 首先尝试从Authorization头获取token
    auth_header = request.headers.get("Authorization", "")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]  # 使用split的maxsplit参数，避免token中有空格时出错
        # 移除可能的双引号（前端可能错误地添加了）
        token = token.strip('"').strip("'")  # 移除单引号和双引号
        if token:
            logger.info(f"✅ 从Authorization头获取token: {token[:30]}..., 路径: {request.url.path}")
            return token
    
    # 如果Authorization头没有token，尝试从Cookie获取
    # 优先使用子路径的Cookie，如果没有则使用根路径的Cookie
    token = request.cookies.get("access_token") or request.cookies.get("access_token_root")
    if token:
        # 移除可能的双引号
        token = token.strip('"').strip("'")
        if token:
            logger.info(f"✅ 从Cookie获取token: {token[:30]}..., Cookie键: {list(request.cookies.keys())}, 路径: {request.url.path}")
            return token
    
    # 详细记录错误信息
    all_cookies = dict(request.cookies)
    logger.error(f"❌ 无法获取token - 路径: {request.url.path}, Authorization头: {auth_header[:100] if auth_header else 'None'}, Cookie数量: {len(all_cookies)}, Cookie键: {list(all_cookies.keys())}, 所有Cookie: {all_cookies}")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )

async def get_current_user(
    request: Request,
    token: str = Depends(get_token_from_request)
) -> UserResponse:
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if global_pool is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="数据库连接池未初始化"
        )
    
    try:
        token_data = verify_token(token)
        if token_data is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception
    
    user_manager = UserManager(global_pool)
    user = await user_manager.get_user_by_username(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_admin_user(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """获取当前管理员用户"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="权限不足：需要管理员权限"
        )
    return current_user

async def get_current_contributor_user(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """获取当前知识库贡献者或管理员用户"""
    logger.info(f"权限检查 - 用户: {current_user.username}, 角色: {current_user.role}, 允许的角色: {[UserRole.CONTRIBUTOR, UserRole.ADMIN]}")
    if current_user.role not in [UserRole.CONTRIBUTOR, UserRole.ADMIN]:
        logger.warning(f"权限检查失败 - 用户: {current_user.username}, 角色: {current_user.role}, 类型: {type(current_user.role)}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="权限不足：需要知识库贡献者或管理员权限"
        )
    logger.info(f"权限检查通过 - 用户: {current_user.username}, 角色: {current_user.role}")
    return current_user

# 创建用户表
async def create_users_table(pool: AsyncConnectionPool):
    """创建用户表"""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            try:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        email VARCHAR(100),
                        is_admin BOOLEAN DEFAULT FALSE,
                        role VARCHAR(20) DEFAULT 'user',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP
                    )
                """)
                
                # 添加role列（如果不存在）
                try:
                    await cur.execute("""
                        ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(20) DEFAULT 'user'
                    """)
                    await conn.commit()
                except Exception:
                    pass  # 列可能已存在
                
                # 迁移现有数据：将is_admin=True的用户（除admin外）改为contributor
                try:
                    # 首先，确保admin用户是admin角色
                    await cur.execute("""
                        UPDATE users 
                        SET role = 'admin', is_admin = TRUE
                        WHERE username = 'admin' AND (role IS NULL OR role != 'admin')
                    """)
                    
                    # 然后，将其他is_admin=True的用户改为contributor
                    await cur.execute("""
                        UPDATE users 
                        SET role = 'contributor', is_admin = FALSE
                        WHERE is_admin = TRUE AND username != 'admin' AND (role IS NULL OR role != 'contributor')
                    """)
                    
                    # 最后，确保所有没有role的用户都设为user
                    await cur.execute("""
                        UPDATE users 
                        SET role = 'user', is_admin = FALSE
                        WHERE role IS NULL OR role = ''
                    """)
                    
                    await conn.commit()
                    logger.info("用户角色数据迁移完成")
                except Exception as e:
                    logger.warning(f"用户角色数据迁移失败（可能已迁移）: {str(e)}")
                await conn.commit()
                logger.info("用户表创建成功")
            except Exception as e:
                logger.error(f"创建用户表失败: {str(e)}")
                raise
