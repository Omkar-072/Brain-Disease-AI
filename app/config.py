"""
Brain Disease AI - Application Configuration
Centralized configuration management using Pydantic Settings
"""
from pydantic_settings import BaseSettings
from pydantic import EmailStr
from typing import List
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application Settings with environment variable support"""
    
    # Application
    APP_NAME: str = "Brain Disease AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-change-in-production-123456789"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = "sqlite:///./brain_disease.db"
    
    # Email Configuration
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    EMAILS_FROM_EMAIL: str = "noreply@braindisease.ai"
    EMAILS_FROM_NAME: str = "Brain Disease AI"
    
    # File Upload
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 10485760  # 10MB
    ALLOWED_EXTENSIONS: str = ".jpg,.jpeg,.png,.dcm,.nii"
    
    # AI Model
    MODEL_PATH: str = "app/ai_models/weights"
    CONFIDENCE_THRESHOLD: float = 0.7
    
    # CORS
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",")]
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Disease Information — matches the 4 trained MRI model classes
DISEASES = {
    "glioma": {
        "name": "Glioma",
        "description": "A type of tumor that occurs in the brain and spinal cord, arising from glial cells.",
        "symptoms": ["Headaches", "Seizures", "Nausea", "Memory loss", "Vision/speech problems"],
        "risk_factors": ["Radiation exposure", "Genetic syndromes", "Age", "Family history"]
    },
    "meningioma": {
        "name": "Meningioma",
        "description": "A tumor that arises from the meninges — the membranes surrounding the brain and spinal cord.",
        "symptoms": ["Headaches", "Vision changes", "Hearing loss", "Memory difficulty", "Seizures"],
        "risk_factors": ["Female sex (hormones)", "Radiation exposure", "Neurofibromatosis type 2", "Older age"]
    },
    "no_tumor": {
        "name": "No Tumor Detected",
        "description": "No abnormal tumor growth was detected in the brain scan.",
        "symptoms": [],
        "risk_factors": []
    },
    "pituitary": {
        "name": "Pituitary Tumor",
        "description": "An abnormal growth in the pituitary gland that can affect hormone production.",
        "symptoms": ["Headaches", "Vision problems", "Hormonal imbalance", "Fatigue", "Mood changes"],
        "risk_factors": ["Genetics (MEN1 syndrome)", "Family history", "Age"]
    },
    "stroke": {
        "name": "Stroke",
        "description": "A medical emergency where poor blood flow to the brain causes cell death.",
        "symptoms": ["Sudden numbness", "Confusion", "Trouble speaking", "Loss of balance", "Severe headache"],
        "risk_factors": ["High blood pressure", "Smoking", "Diabetes", "High cholesterol", "Age"]
    },
    "epilepsy": {
        "name": "Epilepsy",
        "description": "A central nervous system disorder in which brain activity becomes abnormal, causing seizures.",
        "symptoms": ["Uncontrollable jerking movements", "Temporary confusion", "Staring spells", "Loss of consciousness"],
        "risk_factors": ["Head trauma", "Brain conditions", "Infectious diseases", "Prenatal injury"]
    },
    "alzheimer": {
        "name": "Alzheimer's Disease",
        "description": "A progressive disease that destroys memory and other important mental functions.",
        "symptoms": ["Memory loss", "Difficulty thinking and reasoning", "Making judgments", "Personality changes"],
        "risk_factors": ["Age", "Family history", "Genetics", "Head trauma", "Poor heart health"]
    },
    "parkinson": {
        "name": "Parkinson's Disease",
        "description": "A progressive nervous system disorder that affects movement.",
        "symptoms": ["Tremors", "Slowed movement", "Rigid muscles", "Impaired posture", "Speech changes"],
        "risk_factors": ["Age", "Heredity", "Sex (men are more likely)", "Exposure to toxins"]
    },
    "brain_tumor": {
        "name": "General Brain Tumor",
        "description": "A mass or growth of abnormal cells in your brain.",
        "symptoms": ["Persistent headaches", "Vision/hearing problems", "Seizures", "Personality changes"],
        "risk_factors": ["Radiation exposure", "Family history", "Chemical exposure"]
    }
}

# Treatment Information — matches the 4 trained MRI model classes
TREATMENTS = {
    "glioma": {
        "medications": ["Temozolomide (chemotherapy)", "Bevacizumab", "Corticosteroids"],
        "procedures": ["Surgical resection", "Radiation therapy", "Tumor treating fields (TTF)"],
        "lifestyle": ["Regular MRI monitoring", "Cognitive rehabilitation", "Nutrition support", "Support groups"],
        "specialists": ["Neuro-oncologist", "Neurosurgeon", "Radiation oncologist"]
    },
    "meningioma": {
        "medications": ["Hydroxyurea (if inoperable)", "Mifepristone (investigational)", "Pain management"],
        "procedures": ["Surgical removal", "Stereotactic radiosurgery (Gamma Knife)", "Radiation therapy"],
        "lifestyle": ["Regular imaging follow-ups", "Seizure precautions", "Stress management"],
        "specialists": ["Neurosurgeon", "Neurologist", "Radiation oncologist"]
    },
    "no_tumor": {
        "medications": [],
        "procedures": [],
        "lifestyle": ["Maintain healthy diet", "Regular exercise", "Routine check-ups"],
        "specialists": ["Neurologist (for any persisting symptoms)"]
    },
    "pituitary": {
        "medications": ["Dopamine agonists (cabergoline, bromocriptine)", "Somatostatin analogues", "Hormone replacement"],
        "procedures": ["Transsphenoidal surgery", "Radiation therapy", "Radiosurgery"],
        "lifestyle": ["Hormone level monitoring", "Regular endocrinology visits", "Eye exams"],
        "specialists": ["Endocrinologist", "Neurosurgeon", "Ophthalmologist"]
    },
    "stroke": {
        "medications": ["TPA (alteplase)", "Blood thinners", "Statins", "Blood pressure meds"],
        "procedures": ["Emergency endovascular procedures", "Carotid endarterectomy", "Angioplasty"],
        "lifestyle": ["Quit smoking", "Healthy diet", "Exercise", "Limit alcohol"],
        "specialists": ["Neurologist", "Physical Therapist", "Occupational Therapist"]
    },
    "epilepsy": {
        "medications": ["Anti-seizure medications (e.g., Levetiracetam, Lamotrigine)", "Nerve pain medications"],
        "procedures": ["Vagus nerve stimulation", "Resective surgery", "Deep brain stimulation"],
        "lifestyle": ["Ketogenic diet", "Adequate sleep", "Stress management"],
        "specialists": ["Neurologist (Epileptologist)", "Neurosurgeon"]
    },
    "alzheimer": {
        "medications": ["Cholinesterase inhibitors (Donepezil)", "Memantine", "Aducanumab"],
        "procedures": ["Cognitive therapy", "Clinical trials"],
        "lifestyle": ["Physical exercise", "Social engagement", "Nutritious diet", "Mental stimulation"],
        "specialists": ["Neurologist", "Geriatrician", "Psychiatrist"]
    },
    "parkinson": {
        "medications": ["Carbidopa-levodopa", "Dopamine agonists", "MAO B inhibitors"],
        "procedures": ["Deep brain stimulation (DBS)", "Physical therapy"],
        "lifestyle": ["Aerobic exercise", "Massage therapy", "Tai chi / Yoga"],
        "specialists": ["Neurologist (Movement Disorder Specialist)", "Physical Therapist"]
    },
    "brain_tumor": {
        "medications": ["Chemotherapy", "Targeted drug therapy", "Corticosteroids"],
        "procedures": ["Surgery", "Radiation therapy", "Radiosurgery"],
        "lifestyle": ["Palliative care", "Physical therapy", "Nutritional support"],
        "specialists": ["Neuro-oncologist", "Neurosurgeon", "Radiation oncologist"]
    }
}

# Hospital Recommendations (Static Data)
RECOMMENDED_HOSPITALS = [
    {
        "name": "Apollo Hospitals",
        "location": "Multiple locations across India",
        "specialties": ["Neurology", "Neurosurgery", "Oncology"],
        "contact": "+91-1860-500-1066",
        "website": "https://www.apollohospitals.com"
    },
    {
        "name": "AIIMS Delhi",
        "location": "New Delhi, India",
        "specialties": ["Neurology", "Neurosurgery", "Research"],
        "contact": "+91-11-26588500",
        "website": "https://www.aiims.edu"
    },
    {
        "name": "Fortis Healthcare",
        "location": "Multiple locations across India",
        "specialties": ["Brain & Spine", "Neurology", "Oncology"],
        "contact": "+91-8010-994-994",
        "website": "https://www.fortishealthcare.com"
    },
    {
        "name": "Max Healthcare",
        "location": "Delhi NCR, India",
        "specialties": ["Neurosciences", "Oncology", "Critical Care"],
        "contact": "+91-11-2651-5050",
        "website": "https://www.maxhealthcare.in"
    },
    {
        "name": "Manipal Hospitals",
        "location": "Bangalore & other cities",
        "specialties": ["Neurology", "Neurosurgery", "Rehabilitation"],
        "contact": "+91-80-2502-4444",
        "website": "https://www.manipalhospitals.com"
    }
]

# Create settings instance
settings = get_settings()
