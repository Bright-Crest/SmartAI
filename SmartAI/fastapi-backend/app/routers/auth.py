from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Any
import logging
import jwt
from pydantic import ValidationError

from app.core.config import settings
from app.core.dependencies import get_db
from app.models.user import Token, TokenPayload, UserLogin, User

# 为了示例，设置一个SECRET_KEY，实际项目中应在环境变量中配置
SECRET_KEY = "这是一个示例密钥，实际项目中应该使用随机生成的强密钥"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
)

logger = logging.getLogger(__name__)

def create_access_token(*, data: dict, expires_delta: timedelta = None) -> str:
    """创建JWT访问令牌"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def authenticate_user(email: str, password: str, db) -> User:
    """认证用户"""
    # 在实际应用中，应该从数据库查询用户并验证密码
    # 这是一个模拟实现
    if email != "user@example.com" or password != "StrongP@ss1":
        return None
    
    return User(
        id=1,
        email=email,
        username="testuser",
        is_active=True,
        is_admin=False
    )


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db = Depends(get_db)
) -> Any:
    """
    OAuth2兼容登录，获取访问令牌
    """
    logger.info(f"登录尝试: {form_data.username}")
    
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        logger.warning(f"登录失败: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        logger.warning(f"非活动用户尝试登录: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="账户未激活"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires
    )
    
    logger.info(f"登录成功: {user.username}")
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.post("/login/json", response_model=Token)
def login_json(
    user_data: UserLogin,
    db = Depends(get_db)
) -> Any:
    """
    使用JSON格式登录
    """
    logger.info(f"JSON登录尝试: {user_data.email}")
    
    user = authenticate_user(user_data.email, user_data.password, db)
    if not user:
        logger.warning(f"JSON登录失败: {user_data.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="邮箱或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        logger.warning(f"非活动用户尝试JSON登录: {user_data.email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="账户未激活"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires
    )
    
    logger.info(f"JSON登录成功: {user.username}")
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.post("/refresh-token", response_model=Token)
def refresh_token(
    token: str = Depends(OAuth2PasswordRequestForm),
    db = Depends(get_db)
) -> Any:
    """
    刷新访问令牌
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        token_data = TokenPayload(**payload)
        
        if token_data.exp and datetime.fromtimestamp(token_data.exp) < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="令牌已过期",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except (jwt.JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    new_access_token = create_access_token(
        data={"sub": token_data.sub},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": new_access_token,
        "token_type": "bearer"
    } 