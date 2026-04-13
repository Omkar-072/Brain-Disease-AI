"""
Brain Disease AI - Database Models
Complete SQLAlchemy ORM models for the application
"""
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float,
    ForeignKey, Enum, LargeBinary, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from app.database.connection import Base


class UserRole(str, enum.Enum):
    """User role enumeration"""
    PATIENT = "patient"
    DOCTOR = "doctor"
    ADMIN = "admin"


class ScanStatus(str, enum.Enum):
    """Scan processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DiseaseType(str, enum.Enum):
    """Supported disease types for detection.

    All values are stored as UPPERCASE strings so they match the canonical
    labels produced by the predictor and the resolve_disease_enum() helper.
    Never change these values without also updating VALID_DISEASE_TYPES in
    scan_routes.py and LABEL_MAP in predictor.py.
    """
    GLIOMA              = "GLIOMA"
    MENINGIOMA          = "MENINGIOMA"
    NO_TUMOR            = "NO_TUMOR"
    PITUITARY           = "PITUITARY"
    VERY_MILD_DEMENTED  = "VERY_MILD_DEMENTED"
    MILD_DEMENTED       = "MILD_DEMENTED"
    MODERATE_DEMENTED   = "MODERATE_DEMENTED"
    NON_DEMENTED        = "NON_DEMENTED"
    NORMAL              = "NORMAL"          # healthy brain (both models negative)
    INCONCLUSIVE        = "INCONCLUSIVE"
    STROKE              = "STROKE"
    NO_STROKE           = "NO_STROKE"
    PARKINSON           = "PARKINSON"


# ============= USER MODELS =============

class User(Base):
    """User model for authentication and profile management"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Profile Information
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    phone = Column(String(20), nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    gender = Column(String(20), nullable=True)
    address = Column(Text, nullable=True)
    profile_image = Column(String(500), nullable=True)
    
    # Medical Information (for patients)
    blood_group = Column(String(10), nullable=True)
    medical_history = Column(Text, nullable=True)
    emergency_contact = Column(String(100), nullable=True)
    emergency_phone = Column(String(20), nullable=True)
    
    # Role and Status
    role = Column(Enum(UserRole), default=UserRole.PATIENT, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    scans = relationship("BrainScan", back_populates="user", cascade="all, delete-orphan", foreign_keys="[BrainScan.user_id]")
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    password_resets = relationship("PasswordReset", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}')>"


class PasswordReset(Base):
    """Password reset tokens"""
    __tablename__ = "password_resets"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token = Column(String(255), unique=True, index=True, nullable=False)
    otp = Column(String(6), nullable=True)  # 6-digit OTP
    is_used = Column(Boolean, default=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="password_resets")


class LoginHistory(Base):
    """Track user login history"""
    __tablename__ = "login_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    login_time = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(20), default="success")  # success, failed
    
    # Relationship
    user = relationship("User")


# ============= BRAIN SCAN MODELS =============

class BrainScan(Base):
    """Brain scan records with AI analysis results"""
    __tablename__ = "brain_scans"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Scan Information
    scan_type = Column(String(50), nullable=False)  # MRI, CT, PET
    scan_date = Column(DateTime(timezone=True), server_default=func.now())
    file_path = Column(String(500), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=True)  # in bytes
    file_hash = Column(String(64), nullable=True)  # SHA256 hash for integrity
    
    # Processing Status
    status = Column(Enum(ScanStatus), default=ScanStatus.PENDING)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    
    # AI Analysis Results
    predicted_disease = Column(Enum(DiseaseType), nullable=True)
    confidence_score = Column(Float, nullable=True)
    all_predictions = Column(JSON, nullable=True)  # Store all disease probabilities

    # Per-model results (tumor + alzheimer)
    tumor_result = Column(String(100), nullable=True)
    tumor_confidence = Column(Float, nullable=True)
    alz_result = Column(String(100), nullable=True)
    alz_confidence = Column(Float, nullable=True)
    
    # Additional Medical Notes
    notes = Column(Text, nullable=True)
    doctor_review = Column(Text, nullable=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="scans", foreign_keys=[user_id])
    reviewer = relationship("User", foreign_keys=[reviewed_by])
    
    def __repr__(self):
        return f"<BrainScan(id={self.id}, user_id={self.user_id}, disease='{self.predicted_disease}')>"


class ScanReport(Base):
    """Detailed reports generated for scans"""
    __tablename__ = "scan_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(Integer, ForeignKey("brain_scans.id", ondelete="CASCADE"), nullable=False)
    
    # Report Content
    summary = Column(Text, nullable=True)
    detailed_findings = Column(Text, nullable=True)
    recommendations = Column(Text, nullable=True)
    
    # Treatment Suggestions
    suggested_treatments = Column(JSON, nullable=True)
    precautions = Column(JSON, nullable=True)
    lifestyle_guidance = Column(JSON, nullable=True)
    
    # Medical Disclaimer
    disclaimer = Column(Text, default="This AI-generated report is for informational purposes only. "
                                      "Please consult a qualified medical professional for diagnosis and treatment.")
    
    # Report Metadata
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    report_version = Column(String(20), default="1.0")
    
    # Relationship
    scan = relationship("BrainScan")


# ============= CHATBOT MODELS =============

class ChatSession(Base):
    """Chat session tracking"""
    __tablename__ = "chat_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_token = Column(String(255), unique=True, index=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    ended_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """Individual chat messages"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    
    # Message Content
    message_type = Column(String(20), default="text")  # text, image, file
    sender = Column(String(20), nullable=False)  # user, bot
    content = Column(Text, nullable=False)
    
    # Intent Detection (for analytics)
    detected_intent = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    
    # Timestamp
    sent_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    session = relationship("ChatSession", back_populates="messages")


# ============= NOTIFICATION MODEL =============

class Notification(Base):
    """User notifications"""
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Notification Content
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String(50), default="info")  # info, warning, success, error
    
    # Status
    is_read = Column(Boolean, default=False)
    read_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    user = relationship("User", back_populates="notifications")


# ============= SYSTEM MODELS =============

class SystemLog(Base):
    """System activity logging"""
    __tablename__ = "system_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    # Log Details
    action = Column(String(100), nullable=False)
    resource = Column(String(100), nullable=True)
    resource_id = Column(Integer, nullable=True)
    details = Column(JSON, nullable=True)
    ip_address = Column(String(50), nullable=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DiseaseInfo(Base):
    """Disease information database"""
    __tablename__ = "disease_info"
    
    id = Column(Integer, primary_key=True, index=True)
    disease_type = Column(Enum(DiseaseType), unique=True, nullable=False)
    
    # Information
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    symptoms = Column(JSON, nullable=True)
    causes = Column(JSON, nullable=True)
    risk_factors = Column(JSON, nullable=True)
    
    # Treatment Information
    treatments = Column(JSON, nullable=True)
    medications = Column(JSON, nullable=True)
    precautions = Column(JSON, nullable=True)
    lifestyle_changes = Column(JSON, nullable=True)
    
    # Metadata
    last_updated = Column(DateTime(timezone=True), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ContactInquiry(Base):
    """Contact form submissions"""
    __tablename__ = "contact_inquiries"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Contact Information
    name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=False)
    phone = Column(String(20), nullable=True)
    subject = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    
    # Status
    status = Column(String(20), default="pending")  # pending, responded, closed
    responded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    response = Column(Text, nullable=True)
    responded_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
