from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import Any, List
import logging

from app.core.dependencies import get_current_user, get_admin_user, get_db
from app.models.user import User, UserCreate, UserUpdate, Token, UserLogin, UserWithPreferences

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "未找到用户"}},
)

logger = logging.getLogger(__name__)

@router.post("/", response_model=User, status_code=status.HTTP_201_CREATED)
def create_user(user: UserCreate, db = Depends(get_db)) -> Any:
    """
    创建新用户
    """
    logger.info(f"创建用户请求: {user.username}")
    
    # 这里应该检查用户是否已存在并插入数据库
    # 这是模拟实现
    
    return User(
        id=1,
        email=user.email,
        username=user.username,
        is_active=user.is_active,
        is_admin=False
    )


@router.get("/me", response_model=UserWithPreferences)
def read_user_me(current_user: User = Depends(get_current_user)) -> Any:
    """
    获取当前登录用户信息
    """
    logger.info(f"获取当前用户信息: {current_user.username}")
    
    # 在实际应用中，应该从数据库获取完整的用户数据
    return UserWithPreferences(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        is_active=current_user.is_active,
        is_admin=current_user.is_admin,
        preferred_networks=["Ethereum", "Bitcoin"],
        preferred_dashboard_timerange="30d",
        alert_threshold=75
    )


@router.put("/me", response_model=User)
def update_user_me(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db = Depends(get_db)
) -> Any:
    """
    更新当前用户信息
    """
    logger.info(f"更新用户信息请求: {current_user.username}")
    
    # 在实际应用中，应该验证并更新数据库中的用户数据
    updated_user = User(
        id=current_user.id,
        email=user_update.email or current_user.email,
        username=user_update.username or current_user.username,
        is_active=user_update.is_active if user_update.is_active is not None else current_user.is_active,
        is_admin=current_user.is_admin  # 普通用户不能将自己升级为管理员
    )
    
    return updated_user


@router.get("/{user_id}", response_model=User)
def read_user_by_id(
    user_id: int,
    current_user: User = Depends(get_admin_user),
    db = Depends(get_db)
) -> Any:
    """
    根据ID获取用户信息（仅管理员）
    """
    logger.info(f"管理员获取用户信息: ID={user_id}")
    
    # 在实际应用中，应该从数据库查询用户
    if user_id != 1:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    return User(
        id=user_id,
        email="user@example.com",
        username="testuser",
        is_active=True,
        is_admin=False
    )


@router.put("/{user_id}", response_model=User)
def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_user: User = Depends(get_admin_user),
    db = Depends(get_db)
) -> Any:
    """
    管理员更新用户信息
    """
    logger.info(f"管理员更新用户信息: ID={user_id}")
    
    # 在实际应用中，应该验证用户存在并更新数据库
    if user_id != 1:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    updated_user = User(
        id=user_id,
        email=user_update.email or "user@example.com",
        username=user_update.username or "testuser",
        is_active=user_update.is_active if user_update.is_active is not None else True,
        is_admin=user_update.is_admin if user_update.is_admin is not None else False
    )
    
    return updated_user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(
    user_id: int,
    current_user: User = Depends(get_admin_user),
    db = Depends(get_db)
) -> None:
    """
    管理员删除用户
    """
    logger.info(f"管理员删除用户: ID={user_id}")
    
    # 在实际应用中，应该验证用户存在并从数据库删除
    if user_id != 1:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 实际删除操作会在这里进行


@router.get("/", response_model=List[User])
def read_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_admin_user),
    db = Depends(get_db)
) -> Any:
    """
    管理员获取用户列表
    """
    logger.info(f"管理员获取用户列表: skip={skip}, limit={limit}")
    
    # 在实际应用中，应该从数据库查询所有用户
    users = [
        User(
            id=1,
            email="user@example.com",
            username="testuser",
            is_active=True,
            is_admin=False
        ),
        User(
            id=2,
            email="admin@example.com",
            username="adminuser",
            is_active=True,
            is_admin=True
        )
    ]
    
    return users 