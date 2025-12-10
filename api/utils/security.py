"""
FastAPI 鉴权辅助：提供简易 Bearer Token 校验。
"""

import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# 可复用的 Bearer 解析器
bearer_scheme = HTTPBearer(auto_error=False)


def require_bearer_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """
    轻量级 Bearer Token 校验。

    - 未设置环境变量 AIVISUAL_API_TOKEN 时：不强制鉴权（便于开发调试）
    - 设置后：请求需携带 Authorization: Bearer <token>
    """
    expected_token = os.getenv("AIVISUAL_API_TOKEN")
    if not expected_token:
        return True  # no auth configured

    if (
        not credentials
        or credentials.scheme.lower() != "bearer"
        or credentials.credentials != expected_token
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token",
        )

    return True
