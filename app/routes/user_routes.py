"""
Brain Disease AI - User Routes
User profile management and patient data APIs
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional

from app.database import get_db, User, BrainScan, Notification
from app.schemas import (
    UserResponse, UserProfile, UserUpdate, NotificationResponse,
    MessageResponse, DashboardStats, PatientDashboard
)
from app.auth.security import get_current_active_user

router = APIRouter(tags=["Users"])


@router.get("/me/profile", response_model=UserProfile)
async def get_my_profile(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get current user's full profile with scan statistics"""
    # Get scan count
    total_scans = db.query(func.count(BrainScan.id)).filter(
        BrainScan.user_id == current_user.id
    ).scalar()
    
    # Create profile response
    profile_data = UserProfile(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        phone=current_user.phone,
        role=current_user.role,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        date_of_birth=current_user.date_of_birth,
        gender=current_user.gender,
        profile_image=current_user.profile_image,
        created_at=current_user.created_at,
        last_login=current_user.last_login,
        address=current_user.address,
        blood_group=current_user.blood_group,
        medical_history=current_user.medical_history,
        emergency_contact=current_user.emergency_contact,
        emergency_phone=current_user.emergency_phone,
        total_scans=total_scans
    )
    
    return profile_data


@router.put("/me/profile", response_model=UserResponse)
async def update_my_profile(
    update_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update current user's profile"""
    # Update only provided fields
    update_dict = update_data.model_dump(exclude_unset=True)
    
    for field, value in update_dict.items():
        if hasattr(current_user, field):
            setattr(current_user, field, value)
    
    db.commit()
    db.refresh(current_user)
    
    return current_user


@router.get("/me/dashboard", response_model=PatientDashboard)
async def get_dashboard(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get patient dashboard with statistics"""
    from app.database.models import ScanStatus, DiseaseType
    
    # Get scan statistics
    total_scans = db.query(func.count(BrainScan.id)).filter(
        BrainScan.user_id == current_user.id
    ).scalar()
    
    completed_scans = db.query(func.count(BrainScan.id)).filter(
        BrainScan.user_id == current_user.id,
        BrainScan.status == ScanStatus.COMPLETED
    ).scalar()
    
    pending_scans = db.query(func.count(BrainScan.id)).filter(
        BrainScan.user_id == current_user.id,
        BrainScan.status.in_([ScanStatus.PENDING, ScanStatus.PROCESSING])
    ).scalar()
    
    # Get latest diagnosis
    latest_scan = db.query(BrainScan).filter(
        BrainScan.user_id == current_user.id,
        BrainScan.status == ScanStatus.COMPLETED
    ).order_by(BrainScan.processed_at.desc()).first()
    
    # Get disease distribution
    disease_counts = db.query(
        BrainScan.predicted_disease,
        func.count(BrainScan.id)
    ).filter(
        BrainScan.user_id == current_user.id,
        BrainScan.status == ScanStatus.COMPLETED,
        BrainScan.predicted_disease != None
    ).group_by(BrainScan.predicted_disease).all()
    
    disease_distribution = {
        str(disease.value) if disease else "unknown": count 
        for disease, count in disease_counts
    }
    
    # Get recent scans
    recent_scans = db.query(BrainScan).filter(
        BrainScan.user_id == current_user.id
    ).order_by(BrainScan.created_at.desc()).limit(5).all()
    
    # Get unread notifications count
    unread_notifications = db.query(func.count(Notification.id)).filter(
        Notification.user_id == current_user.id,
        Notification.is_read == False
    ).scalar()
    
    # Build dashboard response
    stats = DashboardStats(
        total_scans=total_scans,
        completed_scans=completed_scans,
        pending_scans=pending_scans,
        latest_diagnosis=latest_scan,
        disease_distribution=disease_distribution
    )
    
    return PatientDashboard(
        user=current_user,
        stats=stats,
        recent_scans=recent_scans,
        notifications=unread_notifications
    )


@router.get("/me/notifications", response_model=List[NotificationResponse])
async def get_notifications(
    unread_only: bool = False,
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user notifications"""
    query = db.query(Notification).filter(
        Notification.user_id == current_user.id
    )
    
    if unread_only:
        query = query.filter(Notification.is_read == False)
    
    notifications = query.order_by(
        Notification.created_at.desc()
    ).offset(skip).limit(limit).all()
    
    return notifications


@router.put("/me/notifications/{notification_id}/read", response_model=MessageResponse)
async def mark_notification_read(
    notification_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mark a notification as read"""
    from datetime import datetime
    
    notification = db.query(Notification).filter(
        Notification.id == notification_id,
        Notification.user_id == current_user.id
    ).first()
    
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found"
        )
    
    notification.is_read = True
    notification.read_at = datetime.utcnow()
    db.commit()
    
    return MessageResponse(message="Notification marked as read")


@router.put("/me/notifications/read-all", response_model=MessageResponse)
async def mark_all_notifications_read(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Mark all notifications as read"""
    from datetime import datetime
    
    db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.is_read == False
    ).update({
        "is_read": True,
        "read_at": datetime.utcnow()
    })
    
    db.commit()
    
    return MessageResponse(message="All notifications marked as read")


@router.delete("/me/account", response_model=MessageResponse)
async def deactivate_account(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Deactivate user account (soft delete)"""
    current_user.is_active = False
    db.commit()
    
    return MessageResponse(message="Account deactivated successfully")
