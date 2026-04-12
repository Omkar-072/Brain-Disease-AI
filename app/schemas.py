"""
Brain Disease AI - Pydantic Schemas for Request/Response Validation
"""
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ModelPrediction(BaseModel):
    label: str
    confidence: float

class AllPredictions(BaseModel):
    tumor_model: ModelPrediction
    alz_model: ModelPrediction

# ============= ENUMS =============

class UserRoleEnum(str, Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"
    ADMIN = "admin"


class ScanStatusEnum(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DiseaseTypeEnum(str, Enum):
    GLIOMA             = "GLIOMA"
    MENINGIOMA         = "MENINGIOMA"
    NO_TUMOR           = "NO_TUMOR"
    PITUITARY          = "PITUITARY"
    NON_DEMENTED       = "NON_DEMENTED"
    VERY_MILD_DEMENTED = "VERY_MILD_DEMENTED"
    MILD_DEMENTED      = "MILD_DEMENTED"
    MODERATE_DEMENTED  = "MODERATE_DEMENTED"
    NORMAL             = "NORMAL"
    INCONCLUSIVE       = "INCONCLUSIVE"

# ============= USER SCHEMAS =============

class UserBase(BaseModel):
    """Base user schema"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None


class UserCreate(UserBase):
    """Schema for user registration"""
    password: str = Field(..., min_length=8, max_length=100)
    confirm_password: str = Field(..., min_length=8, max_length=100)
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr
    password: str


class UserUpdate(BaseModel):
    """Schema for updating user profile"""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = None
    address: Optional[str] = None
    blood_group: Optional[str] = None
    medical_history: Optional[str] = None
    emergency_contact: Optional[str] = None
    emergency_phone: Optional[str] = None


class UserResponse(UserBase):
    """Schema for user response"""
    id: int
    role: UserRoleEnum
    is_active: bool
    is_verified: bool
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = None
    profile_image: Optional[str] = None
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserProfile(UserResponse):
    """Extended user profile"""
    address: Optional[str] = None
    blood_group: Optional[str] = None
    medical_history: Optional[str] = None
    emergency_contact: Optional[str] = None
    emergency_phone: Optional[str] = None
    total_scans: int = 0


# ============= TOKEN SCHEMAS =============

class Token(BaseModel):
    """Schema for JWT tokens"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Schema for token payload data"""
    user_id: Optional[int] = None
    email: Optional[str] = None
    role: Optional[str] = None


class RefreshToken(BaseModel):
    """Schema for refresh token request"""
    refresh_token: str


# ============= PASSWORD SCHEMAS =============

class PasswordResetRequest(BaseModel):
    """Schema for requesting password reset"""
    email: EmailStr


class PasswordResetVerify(BaseModel):
    """Schema for verifying OTP"""
    email: EmailStr
    otp: str = Field(..., min_length=6, max_length=6)


class PasswordResetConfirm(BaseModel):
    """Schema for confirming password reset"""
    token: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class ChangePassword(BaseModel):
    """Schema for changing password"""
    current_password: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


# ============= SCAN SCHEMAS =============

class ScanBase(BaseModel):
    """Base scan schema"""
    scan_type: str = Field(..., description="Type of scan: MRI, CT, PET")
    notes: Optional[str] = None


class ScanCreate(ScanBase):
    """Schema for creating a scan"""
    pass


class ScanResponse(ScanBase):
    """Schema for scan response"""
    id: int
    user_id: int
    file_name: str
    file_size: Optional[int] = None
    status: ScanStatusEnum
    predicted_disease: Optional[DiseaseTypeEnum] = None
    confidence_score: Optional[float] = None
    # Relaxed to plain dict so any valid JSON blob from the DB is accepted
    all_predictions: Optional[Dict[str, Any]] = None
    scan_date: datetime
    processed_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ScanDetail(ScanResponse):
    """Detailed scan response with additional info"""
    doctor_review: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    report: Optional['ScanReportResponse'] = None


class ScanReportResponse(BaseModel):
    """Schema for scan report"""
    id: int
    summary: Optional[str] = None
    detailed_findings: Optional[str] = None
    recommendations: Optional[str] = None
    suggested_treatments: Optional[List[str]] = None
    precautions: Optional[List[str]] = None
    lifestyle_guidance: Optional[List[str]] = None
    disclaimer: str
    generated_at: datetime
    
    class Config:
        from_attributes = True


# ============= AI PREDICTION SCHEMAS =============

class PredictionResult(BaseModel):
    """Schema for AI prediction result"""
    scan_id: int
    predicted_disease: str
    confidence_score: float
    all_predictions: Dict[str, float]
    disclaimer: str = "This AI-generated prediction is for informational purposes only. Please consult a qualified medical professional for diagnosis and treatment."


class PredictionHistory(BaseModel):
    """Schema for prediction history"""
    total_scans: int
    predictions: List[ScanResponse]


# ============= CHATBOT SCHEMAS =============

class ChatMessageCreate(BaseModel):
    """Schema for creating chat message"""
    message: str = Field(..., min_length=1, max_length=1000)


class ChatMessageResponse(BaseModel):
    """Schema for chat message response"""
    id: int
    sender: str
    content: str
    detected_intent: Optional[str] = None
    sent_at: datetime
    
    class Config:
        from_attributes = True


class ChatSessionResponse(BaseModel):
    """Schema for chat session"""
    session_id: int
    session_token: str
    started_at: datetime
    messages: List[ChatMessageResponse] = []
    
    class Config:
        from_attributes = True


class ChatBotResponse(BaseModel):
    """Schema for chatbot response"""
    response: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    suggestions: Optional[List[str]] = None


# ============= DASHBOARD SCHEMAS =============

class DashboardStats(BaseModel):
    """Schema for dashboard statistics"""
    total_scans: int
    completed_scans: int
    pending_scans: int
    latest_diagnosis: Optional[ScanResponse] = None
    disease_distribution: Dict[str, int] = {}


class PatientDashboard(BaseModel):
    """Schema for patient dashboard"""
    user: UserResponse
    stats: DashboardStats
    recent_scans: List[ScanResponse] = []
    notifications: int = 0


# ============= DISEASE INFO SCHEMAS =============

class DiseaseInfoResponse(BaseModel):
    """Schema for disease information"""
    disease_type: str
    name: str
    description: str
    symptoms: List[str]
    causes: Optional[List[str]] = None
    risk_factors: List[str]
    treatments: Optional[Dict[str, Any]] = None


class TreatmentInfo(BaseModel):
    """Schema for treatment information"""
    disease: str
    medications: List[str]
    procedures: List[str]
    lifestyle: List[str]
    specialists: List[str]


# ============= CONTACT SCHEMAS =============

class ContactCreate(BaseModel):
    """Schema for contact form"""
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    phone: Optional[str] = None
    subject: str = Field(..., min_length=5, max_length=255)
    message: str = Field(..., min_length=10, max_length=2000)


class ContactResponse(ContactCreate):
    """Schema for contact response"""
    id: int
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============= NOTIFICATION SCHEMAS =============

class NotificationResponse(BaseModel):
    """Schema for notification"""
    id: int
    title: str
    message: str
    notification_type: str
    is_read: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============= HOSPITAL SCHEMAS =============

class HospitalInfo(BaseModel):
    """Schema for hospital information"""
    name: str
    location: str
    specialties: List[str]
    contact: str
    website: str


# ============= GENERAL RESPONSE SCHEMAS =============

class MessageResponse(BaseModel):
    """Generic message response"""
    message: str
    success: bool = True


class ErrorResponse(BaseModel):
    """Error response schema"""
    detail: str
    error_code: Optional[str] = None


class PaginatedResponse(BaseModel):
    """Paginated response schema"""
    total: int
    page: int
    per_page: int
    pages: int
    items: List[Any]
