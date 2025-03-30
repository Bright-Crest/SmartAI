from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, validator
import re


class UserBase(BaseModel):
    """用户基础数据模型"""
    email: Optional[EmailStr] = None
    username: str = Field(..., min_length=3, max_length=50)
    is_active: bool = True
    is_admin: bool = False


class UserCreate(UserBase):
    """用户创建数据模型"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    
    @validator('password')
    def password_complexity(cls, v):
        """验证密码复杂度"""
        if not re.search(r'[A-Z]', v):
            raise ValueError('密码必须包含至少一个大写字母')
        if not re.search(r'[a-z]', v):
            raise ValueError('密码必须包含至少一个小写字母')
        if not re.search(r'[0-9]', v):
            raise ValueError('密码必须包含至少一个数字')
        if not re.search(r'[^A-Za-z0-9]', v):
            raise ValueError('密码必须包含至少一个特殊字符')
        return v


class UserUpdate(BaseModel):
    """用户更新数据模型"""
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    
    @validator('username')
    def username_length(cls, v):
        if v is not None and (len(v) < 3 or len(v) > 50):
            raise ValueError('用户名长度必须在3到50个字符之间')
        return v
    
    @validator('password')
    def password_complexity(cls, v):
        if v is not None:
            if len(v) < 8 or len(v) > 100:
                raise ValueError('密码长度必须在8到100个字符之间')
            if not re.search(r'[A-Z]', v):
                raise ValueError('密码必须包含至少一个大写字母')
            if not re.search(r'[a-z]', v):
                raise ValueError('密码必须包含至少一个小写字母')
            if not re.search(r'[0-9]', v):
                raise ValueError('密码必须包含至少一个数字')
            if not re.search(r'[^A-Za-z0-9]', v):
                raise ValueError('密码必须包含至少一个特殊字符')
        return v


class User(UserBase):
    """用户完整数据模型"""
    id: int
    
    class Config:
        orm_mode = True


class Token(BaseModel):
    """令牌数据模型"""
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    """令牌载荷数据模型"""
    sub: Optional[int] = None
    exp: Optional[int] = None


class UserLogin(BaseModel):
    """用户登录数据模型"""
    email: EmailStr
    password: str


class UserWithPreferences(User):
    """带用户偏好的用户数据模型"""
    preferred_networks: List[str] = []
    preferred_dashboard_timerange: str = "7d"
    alert_threshold: int = 70
    
    class Config:
        orm_mode = True 