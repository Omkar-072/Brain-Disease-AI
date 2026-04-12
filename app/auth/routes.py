"""
Brain Disease AI - Authentication Routes
User registration, login, password reset, and token management
"""
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional

from app.database import get_db, User, UserRole, PasswordReset, LoginHistory
from app.schemas import (
    UserCreate, UserLogin, UserResponse, Token, RefreshToken,
    PasswordResetRequest, PasswordResetVerify, PasswordResetConfirm,
    ChangePassword, MessageResponse
)
from app.auth.security import (
    get_password_hash, verify_password, create_access_token,
    create_refresh_token, verify_token, generate_otp, generate_reset_token,
    get_current_user, get_current_active_user, get_current_user_flexible, rate_limiter
)
from app.services.email_service import send_login_notification, send_password_reset_email
from app.config import get_settings

settings = get_settings()
router = APIRouter(tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    Register a new user account.
    
    - **email**: Valid email address (unique)
    - **username**: Unique username (3-100 characters)
    - **password**: Strong password (min 8 chars, must include uppercase, lowercase, digit)
    """
    # Check rate limiting
    if not rate_limiter.is_allowed(f"register_{user_data.email}", max_requests=3, window_seconds=300):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many registration attempts. Please try again later."
        )
    
    # Check if email already exists
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username already exists
    existing_username = db.query(User).filter(User.username == user_data.username).first()
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    
    new_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        phone=user_data.phone,
        role=UserRole.PATIENT,
        is_active=True,
        is_verified=False  # Will be verified via email
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user


@router.post("/login", response_model=Token)
async def login(
    user_credentials: UserLogin,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Authenticate user and return JWT tokens.
    
    - **email**: User's email address
    - **password**: User's password
    
    Returns access token and refresh token on successful authentication.
    """
    # Check rate limiting
    if not rate_limiter.is_allowed(f"login_{user_credentials.email}", max_requests=5, window_seconds=300):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again later."
        )
    
    # Find user by email
    user = db.query(User).filter(User.email == user_credentials.email).first()
    
    if not user:
        # Log failed attempt
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Verify password
    if not verify_password(user_credentials.password, user.hashed_password):
        # Log failed login attempt
        login_record = LoginHistory(
            user_id=user.id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            status="failed"
        )
        db.add(login_record)
        db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Please contact support."
        )
    
    # Create tokens
    token_data = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role.value
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    # Update last login
    user.last_login = datetime.utcnow()

    # Log successful login
    login_record = LoginHistory(
        user_id=user.id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        status="success"
    )
    db.add(login_record)
    db.commit()

    # Send login notification email in background
    background_tasks.add_task(
        send_login_notification,
        user.email,
        user.first_name or user.username,
        request.client.host if request.client else "Unknown",
        datetime.utcnow().isoformat()
    )

    # Build JSON response and also set an HttpOnly cookie so the browser
    # can load protected images directly (e.g. <img src="/api/v1/scans/1/image">)
    response = JSONResponse(content={
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    })
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,         # JS cannot read it — XSS protection
        samesite="lax",        # CSRF protection for same-origin navigation
        secure=not settings.DEBUG,  # HTTPS only in production
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )
    return response


@router.post("/refresh", response_model=Token)
async def refresh_token(
    token_data: RefreshToken,
    db: Session = Depends(get_db)
):
    """
    Refresh access token using a valid refresh token.
    
    - **refresh_token**: Valid refresh token obtained during login
    """
    # Verify refresh token
    token_payload = verify_token(token_data.refresh_token, "refresh")
    
    if token_payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Get user
    user = db.query(User).filter(User.id == token_payload.user_id).first()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new tokens
    new_token_data = {
        "sub": str(user.id),
        "email": user.email,
        "role": user.role.value
    }
    
    access_token = create_access_token(new_token_data)
    refresh_token = create_refresh_token(new_token_data)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )


@router.post("/password-reset/request", response_model=MessageResponse)
async def request_password_reset(
    reset_request: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Request a password reset. Sends OTP to user's email.
    
    - **email**: Registered email address
    """
    # Rate limiting
    if not rate_limiter.is_allowed(f"reset_{reset_request.email}", max_requests=3, window_seconds=600):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many reset requests. Please try again later."
        )
    
    # Find user
    user = db.query(User).filter(User.email == reset_request.email).first()
    
    # Always return success to prevent email enumeration
    if not user:
        return MessageResponse(message="If the email exists, a password reset OTP has been sent.")
    
    # Generate OTP and token
    otp = generate_otp()
    reset_token = generate_reset_token()
    
    # Invalidate previous reset requests
    db.query(PasswordReset).filter(
        PasswordReset.user_id == user.id,
        PasswordReset.is_used == False
    ).update({"is_used": True})
    
    # Create new password reset record
    password_reset = PasswordReset(
        user_id=user.id,
        token=reset_token,
        otp=otp,
        expires_at=datetime.utcnow() + timedelta(minutes=15)
    )
    
    db.add(password_reset)
    db.commit()
    
    # Send email in background
    background_tasks.add_task(
        send_password_reset_email,
        user.email,
        user.first_name or user.username,
        otp,
        reset_token
    )
    
    return MessageResponse(message="If the email exists, a password reset OTP has been sent.")


@router.post("/password-reset/verify", response_model=MessageResponse)
async def verify_password_reset_otp(
    verify_data: PasswordResetVerify,
    db: Session = Depends(get_db)
):
    """
    Verify OTP for password reset.
    
    - **email**: User's email address
    - **otp**: 6-digit OTP received via email
    """
    # Find user and valid reset request
    user = db.query(User).filter(User.email == verify_data.email).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email or OTP"
        )
    
    reset_record = db.query(PasswordReset).filter(
        PasswordReset.user_id == user.id,
        PasswordReset.otp == verify_data.otp,
        PasswordReset.is_used == False,
        PasswordReset.expires_at > datetime.utcnow()
    ).first()
    
    if not reset_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )
    
    return MessageResponse(
        message="OTP verified successfully",
        success=True
    )


@router.post("/password-reset/confirm", response_model=MessageResponse)
async def confirm_password_reset(
    reset_data: PasswordResetConfirm,
    db: Session = Depends(get_db)
):
    """
    Reset password using verified token.
    
    - **token**: Reset token from the verification step
    - **new_password**: New password (min 8 characters)
    - **confirm_password**: Confirm new password
    """
    # Find valid reset request
    reset_record = db.query(PasswordReset).filter(
        PasswordReset.token == reset_data.token,
        PasswordReset.is_used == False,
        PasswordReset.expires_at > datetime.utcnow()
    ).first()
    
    if not reset_record:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    # Get user and update password
    user = db.query(User).filter(User.id == reset_record.user_id).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update password
    user.hashed_password = get_password_hash(reset_data.new_password)
    
    # Mark reset as used
    reset_record.is_used = True
    
    db.commit()
    
    return MessageResponse(message="Password has been reset successfully")


@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    password_data: ChangePassword,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Change password for authenticated user.
    
    - **current_password**: Current password
    - **new_password**: New password (min 8 characters)
    - **confirm_password**: Confirm new password
    """
    # Verify current password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Check if new password is same as current
    if verify_password(password_data.new_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from current password"
        )
    
    # Update password
    current_user.hashed_password = get_password_hash(password_data.new_password)
    db.commit()
    
    return MessageResponse(message="Password changed successfully")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current authenticated user's information.
    """
    return current_user


@router.post("/logout", response_model=MessageResponse)
async def logout(
    current_user: User = Depends(get_current_active_user)
):
    """
    Logout current user.
    Note: JWT tokens are stateless, so this endpoint is primarily for client-side cleanup.
    In production, implement token blacklisting for complete security.
    """
    # In production, add token to blacklist here
    return MessageResponse(message="Logged out successfully")
