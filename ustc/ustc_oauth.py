"""
USTC 统一身份认证 OAuth2.0 对接模块
基于 https://id.ustc.edu.cn/doc/developer/
"""

import httpx
import secrets
from urllib.parse import urlencode, parse_qs
from typing import Optional, Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class USTCOAuth:
    """USTC统一身份认证OAuth2.0客户端"""
    
    # USTC 统一身份认证端点
    AUTHORIZE_URL = "https://id.ustc.edu.cn/cas/oauth2.0/authorize"
    TOKEN_URL = "https://id.ustc.edu.cn/cas/oauth2.0/accessToken"
    PROFILE_URL = "https://id.ustc.edu.cn/cas/oauth2.0/profile"
    
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        """
        初始化USTC OAuth客户端
        
        Args:
            client_id: 应用的Client ID（由系统管理员提供）
            client_secret: 应用的Client Secret（由系统管理员提供）
            redirect_uri: 授权完成后的回调地址（需要URL编码）
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
    
    def get_authorize_url(self, state: Optional[str] = None) -> tuple[str, str]:
        """
        生成授权URL
        
        Args:
            state: 随机生成的字符串，用于防止CSRF攻击。如果为None，会自动生成
        
        Returns:
            (authorize_url, state) 元组
        """
        if state is None:
            state = secrets.token_urlsafe(32)
        
        params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'state': state
        }
        
        authorize_url = f"{self.AUTHORIZE_URL}?{urlencode(params)}"
        return authorize_url, state
    
    async def get_access_token(self, code: str) -> Dict[str, Any]:
        """
        使用授权码换取access_token
        
        Args:
            code: 从回调URL中获取的授权码
        
        Returns:
            包含access_token、token_type、expires_in的字典
        """
        async with httpx.AsyncClient() as client:
            data = {
                'grant_type': 'authorization_code',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'redirect_uri': self.redirect_uri,
                'code': code
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            try:
                response = await client.post(
                    self.TOKEN_URL,
                    data=data,
                    headers=headers,
                    timeout=10.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"获取access_token失败: {e}")
                raise Exception(f"获取access_token失败: {str(e)}")
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """
        使用access_token获取用户信息
        
        Args:
            access_token: 访问令牌
        
        Returns:
            用户信息字典，包含：
            - id: 用户名
            - attributes: 用户属性（name, email, gid, deptCode等）
            - client_id: 应用ID
        """
        async with httpx.AsyncClient() as client:
            data = {
                'access_token': access_token
            }
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            try:
                response = await client.post(
                    self.PROFILE_URL,
                    data=data,
                    headers=headers,
                    timeout=10.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                logger.error(f"获取用户信息失败: {e}")
                raise Exception(f"获取用户信息失败: {str(e)}")
    
    def parse_user_info(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析用户信息，提取常用字段
        
        Args:
            user_data: 从get_user_info返回的原始数据
        
        Returns:
            解析后的用户信息字典，包含：
            - username: 用户名（使用gid或id）
            - name: 姓名
            - email: 邮箱
            - gid: GID
            - dept_code: 部门编码
            - student_id: 学工号（zjhm）
            - ryfldm: 人员类型代码
            - xbm: 性别码（1=男，2=女）
            - ryzxztdm: 在校状态码
        """
        attributes = user_data.get('attributes', {})
        
        return {
            'username': attributes.get('gid') or user_data.get('id', ''),
            'name': attributes.get('name', ''),
            'email': attributes.get('email', ''),
            'gid': attributes.get('gid', ''),
            'dept_code': attributes.get('deptCode', ''),
            'student_id': attributes.get('zjhm', ''),
            'ryfldm': attributes.get('ryfldm', ''),
            'xbm': attributes.get('xbm', ''),
            'ryzxztdm': attributes.get('ryzxztdm', ''),
            'login_ip': attributes.get('loginip', ''),
            'login_time': attributes.get('logintime', ''),
            'object_id': attributes.get('objectId', ''),
        }

