"""
NSRL CAS 2.0 统一身份认证对接模块
基于CAS 2.0协议
"""
import httpx
import secrets
from urllib.parse import urlencode, parse_qs, urlparse
from typing import Optional, Dict, Any
import logging
from xml.etree import ElementTree as ET
from datetime import datetime

logger = logging.getLogger(__name__)

class NSRLCAS:
    """NSRL CAS 2.0客户端"""
    
    # CAS服务器端点
    LOGIN_URL = "https://nsrloa.ustc.edu.cn/cas/login"
    LOGOUT_URL = "https://nsrloa.ustc.edu.cn/cas/logout"
    SERVICE_VALIDATE_URL = "https://nsrloa.ustc.edu.cn/cas/serviceValidate"
    PROXY_VALIDATE_URL = "https://nsrloa.ustc.edu.cn/cas/proxyValidate"
    
    def __init__(self, service_url: str):
        """
        初始化NSRL CAS客户端
        
        Args:
            service_url: 服务回调URL（需要URL编码）
        """
        self.service_url = service_url
    
    def get_login_url(self, state: Optional[str] = None) -> tuple[str, str]:
        """
        生成CAS登录URL
        
        Args:
            state: 随机生成的字符串，用于防止CSRF攻击。如果为None，会自动生成
        
        Returns:
            (login_url, state) 元组
        """
        if state is None:
            state = secrets.token_urlsafe(32)
        
        params = {
            'service': self.service_url
        }
        
        login_url = f"{self.LOGIN_URL}?{urlencode(params)}"
        return login_url, state
    
    async def validate_ticket(self, ticket: str) -> Dict[str, Any]:
        """
        使用ticket验证并获取用户信息
        
        Args:
            ticket: 从回调URL中获取的ticket
        
        Returns:
            包含用户信息的字典，格式：
            {
                'success': True,
                'username': '用户名',
                'attributes': {
                    'displayName': '显示名称',
                    'mail': '邮箱',
                    'clientIpAddress': 'IP地址',
                    'isFromNewLogin': True/False,
                    'authenticationDate': '认证时间'
                }
            }
            如果验证失败，返回：
            {
                'success': False,
                'error': '错误信息'
            }
        """
        async with httpx.AsyncClient() as client:
            params = {
                'service': self.service_url,
                'ticket': ticket
            }
            
            try:
                response = await client.get(
                    self.SERVICE_VALIDATE_URL,
                    params=params,
                    timeout=10.0
                )
                response.raise_for_status()
                
                # 解析XML响应
                return self._parse_cas_response(response.text)
                
            except httpx.HTTPError as e:
                logger.error(f"验证ticket失败: {e}")
                return {
                    'success': False,
                    'error': f"验证ticket失败: {str(e)}"
                }
            except Exception as e:
                logger.error(f"解析CAS响应失败: {e}")
                return {
                    'success': False,
                    'error': f"解析CAS响应失败: {str(e)}"
                }
    
    def _parse_cas_response(self, xml_text: str) -> Dict[str, Any]:
        """
        解析CAS XML响应
        
        Args:
            xml_text: CAS服务器返回的XML文本
        
        Returns:
            解析后的用户信息字典
        """
        try:
            root = ET.fromstring(xml_text)
            
            # 查找命名空间
            ns = {'cas': 'http://www.yale.edu/tp/cas'}
            
            # 检查是否认证成功
            auth_success = root.find('cas:authenticationSuccess', ns)
            if auth_success is not None:
                # 获取用户名
                user_elem = auth_success.find('cas:user', ns)
                username = user_elem.text if user_elem is not None else ''
                
                # 获取属性
                attributes = {}
                attrs_elem = auth_success.find('cas:attributes', ns)
                if attrs_elem is not None:
                    for attr in attrs_elem:
                        attr_name = attr.tag.split('}')[-1] if '}' in attr.tag else attr.tag
                        attr_value = attr.text if attr.text else ''
                        attributes[attr_name] = attr_value
                
                return {
                    'success': True,
                    'username': username,
                    'attributes': attributes
                }
            
            # 检查是否认证失败
            auth_failure = root.find('cas:authenticationFailure', ns)
            if auth_failure is not None:
                error_code = auth_failure.get('code', 'UNKNOWN_ERROR')
                error_msg = auth_failure.text if auth_failure.text else '认证失败'
                return {
                    'success': False,
                    'error': error_msg,
                    'error_code': error_code
                }
            
            return {
                'success': False,
                'error': '未知的CAS响应格式'
            }
            
        except ET.ParseError as e:
            logger.error(f"XML解析错误: {e}")
            return {
                'success': False,
                'error': f'XML解析错误: {str(e)}'
            }
        except Exception as e:
            logger.error(f"解析CAS响应时发生错误: {e}")
            return {
                'success': False,
                'error': f'解析错误: {str(e)}'
            }
    
    def parse_user_info(self, cas_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析CAS返回的用户信息，提取常用字段
        
        Args:
            cas_data: 从validate_ticket返回的数据
        
        Returns:
            解析后的用户信息字典，包含：
            - username: 用户名（邮箱格式）
            - name: 显示名称
            - email: 邮箱
            - gid: 如果有的话（从username或其他属性提取）
            - client_ip: 客户端IP
            - is_new_login: 是否为新登录
            - auth_date: 认证时间
        """
        if not cas_data.get('success'):
            return {}
        
        attributes = cas_data.get('attributes', {})
        username = cas_data.get('username', '')
        
        # 尝试从username中提取GID（如果格式是gid@mail.ustc.edu.cn）
        gid = ''
        if '@' in username:
            gid = username.split('@')[0]
        
        return {
            'username': username,
            'name': attributes.get('displayName', ''),
            'email': attributes.get('mail', username),
            'gid': gid,
            'client_ip': attributes.get('clientIpAddress', ''),
            'is_new_login': attributes.get('isFromNewLogin', 'false').lower() == 'true',
            'auth_date': attributes.get('authenticationDate', ''),
        }

