"""
Brain Disease AI - Info Routes
Disease information, treatments, hospitals, and system info APIs
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import get_db
from app.schemas import (
    DiseaseInfoResponse, TreatmentInfo, HospitalInfo,
    ContactCreate, ContactResponse, MessageResponse
)
from app.config import DISEASES, TREATMENTS, RECOMMENDED_HOSPITALS, get_settings
from app.database.models import ContactInquiry

router = APIRouter(tags=["Information"])
settings = get_settings()


@router.get("/system")
async def get_system_info():
    """Get system overview and information"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "AI-powered early brain disease detection system",
        "features": [
            "Brain scan analysis using deep learning CNN models",
            "Detection of 5 major brain diseases",
            "Secure patient data management",
            "AI-powered chatbot support",
            "Treatment recommendations and guidance"
        ],
        "supported_diseases": [
            "Stroke",
            "Epilepsy", 
            "Alzheimer's Disease",
            "Parkinson's Disease",
            "Brain Tumor"
        ],
        "supported_scan_types": ["MRI", "CT", "PET"],
        "disclaimer": "This system is for informational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment."
    }


@router.get("/diseases", response_model=List[DiseaseInfoResponse])
async def get_all_diseases():
    """Get information about all supported diseases"""
    diseases = []
    for key, info in DISEASES.items():
        diseases.append(DiseaseInfoResponse(
            disease_type=key,
            name=info["name"],
            description=info["description"],
            symptoms=info["symptoms"],
            risk_factors=info["risk_factors"]
        ))
    return diseases


@router.get("/diseases/{disease_type}", response_model=DiseaseInfoResponse)
async def get_disease_info(disease_type: str):
    """
    Get detailed information about a specific disease.
    
    - **disease_type**: One of: stroke, epilepsy, alzheimer, parkinson, brain_tumor
    """
    disease_key = disease_type.lower().replace(" ", "_")
    
    if disease_key not in DISEASES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Disease not found. Available: {', '.join(DISEASES.keys())}"
        )
    
    info = DISEASES[disease_key]
    treatment = TREATMENTS.get(disease_key, {})
    
    return DiseaseInfoResponse(
        disease_type=disease_key,
        name=info["name"],
        description=info["description"],
        symptoms=info["symptoms"],
        risk_factors=info["risk_factors"],
        treatments={
            "medications": treatment.get("medications", []),
            "procedures": treatment.get("procedures", []),
            "lifestyle": treatment.get("lifestyle", []),
            "specialists": treatment.get("specialists", [])
        }
    )


@router.get("/treatments/{disease_type}", response_model=TreatmentInfo)
async def get_treatment_info(disease_type: str):
    """
    Get treatment information for a specific disease.
    
    - **disease_type**: One of: stroke, epilepsy, alzheimer, parkinson, brain_tumor
    """
    disease_key = disease_type.lower().replace(" ", "_")
    
    if disease_key not in TREATMENTS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Treatment info not found. Available: {', '.join(TREATMENTS.keys())}"
        )
    
    treatment = TREATMENTS[disease_key]
    
    return TreatmentInfo(
        disease=disease_key,
        medications=treatment.get("medications", []),
        procedures=treatment.get("procedures", []),
        lifestyle=treatment.get("lifestyle", []),
        specialists=treatment.get("specialists", [])
    )


@router.get("/precautions/{disease_type}")
async def get_precautions(disease_type: str):
    """Get precautions and lifestyle guidance for a disease"""
    disease_key = disease_type.lower().replace(" ", "_")
    
    if disease_key not in DISEASES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Disease not found"
        )
    
    disease = DISEASES[disease_key]
    treatment = TREATMENTS.get(disease_key, {})
    
    return {
        "disease": disease["name"],
        "risk_factors_to_avoid": disease.get("risk_factors", []),
        "lifestyle_recommendations": treatment.get("lifestyle", []),
        "general_precautions": [
            "Follow your doctor's advice strictly",
            "Take medications as prescribed",
            "Attend regular check-ups",
            "Maintain a healthy diet and exercise routine",
            "Manage stress effectively",
            "Get adequate sleep",
            "Stay hydrated",
            "Avoid smoking and excessive alcohol"
        ],
        "warning_signs": disease.get("symptoms", []),
        "emergency_note": "Seek immediate medical attention if symptoms worsen or new symptoms appear."
    }


@router.get("/hospitals", response_model=List[HospitalInfo])
async def get_recommended_hospitals():
    """Get list of recommended hospitals and specialists"""
    return [HospitalInfo(**hospital) for hospital in RECOMMENDED_HOSPITALS]


@router.get("/hospitals/{specialty}")
async def get_hospitals_by_specialty(specialty: str):
    """Get hospitals filtered by specialty"""
    specialty_lower = specialty.lower()
    
    filtered = [
        hospital for hospital in RECOMMENDED_HOSPITALS
        if any(specialty_lower in s.lower() for s in hospital["specialties"])
    ]
    
    if not filtered:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No hospitals found for the given specialty"
        )
    
    return [HospitalInfo(**hospital) for hospital in filtered]


@router.post("/contact", response_model=ContactResponse, status_code=status.HTTP_201_CREATED)
async def submit_contact_form(
    contact: ContactCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Submit a contact inquiry"""
    inquiry = ContactInquiry(
        name=contact.name,
        email=contact.email,
        phone=contact.phone,
        subject=contact.subject,
        message=contact.message
    )
    
    db.add(inquiry)
    db.commit()
    db.refresh(inquiry)
    
    # Send email notification to admin in background
    from app.services.email_service import send_contact_notification
    background_tasks.add_task(
        send_contact_notification,
        name=contact.name,
        email=contact.email,
        subject=contact.subject,
        message=contact.message,
        phone=contact.phone
    )
    
    return inquiry


@router.get("/services")
async def get_services():
    """Get list of services offered"""
    return {
        "services": [
            {
                "name": "AI Brain Scan Analysis",
                "description": "Upload MRI/CT scans for instant AI-powered disease detection",
                "features": ["Fast processing", "5 disease detection", "Confidence scores"]
            },
            {
                "name": "Disease Information",
                "description": "Comprehensive information about brain diseases",
                "features": ["Symptoms", "Causes", "Risk factors", "Prevention"]
            },
            {
                "name": "Treatment Guidance",
                "description": "AI-recommended treatments and specialist referrals",
                "features": ["Medications", "Procedures", "Lifestyle changes"]
            },
            {
                "name": "AI Chatbot Support",
                "description": "24/7 AI assistant for health queries",
                "features": ["Instant responses", "Disease FAQs", "Symptom checker"]
            },
            {
                "name": "Patient Dashboard",
                "description": "Track your scan history and health metrics",
                "features": ["Scan history", "Reports", "Notifications"]
            },
            {
                "name": "Hospital Recommendations",
                "description": "Find specialists and hospitals near you",
                "features": ["Top neurologists", "Multi-specialty hospitals", "Contact info"]
            }
        ]
    }


@router.get("/faq")
async def get_faq():
    """Get frequently asked questions"""
    return {
        "faqs": [
            {
                "question": "How accurate is the AI diagnosis?",
                "answer": "Our AI model has been trained on thousands of brain scans and achieves high accuracy. However, it should be used as a screening tool, not a definitive diagnosis. Always consult a medical professional."
            },
            {
                "question": "What file formats are supported?",
                "answer": "We support JPEG, PNG, DICOM (.dcm), and NIfTI (.nii) formats for brain scans."
            },
            {
                "question": "How long does the analysis take?",
                "answer": "Most scans are processed within 1-2 minutes. Complex scans may take slightly longer."
            },
            {
                "question": "Is my medical data secure?",
                "answer": "Yes, all data is encrypted and stored securely. We follow strict privacy guidelines and never share your data without consent."
            },
            {
                "question": "Can I delete my scan history?",
                "answer": "Yes, you can delete individual scans or your entire account from the dashboard."
            },
            {
                "question": "What diseases can the system detect?",
                "answer": "Currently, we can detect Stroke, Epilepsy, Alzheimer's Disease, Parkinson's Disease, and Brain Tumors."
            },
            {
                "question": "Should I rely solely on this AI diagnosis?",
                "answer": "No. This tool is meant to assist, not replace, professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment."
            }
        ]
    }


@router.get("/disclaimer")
async def get_medical_disclaimer():
    """Get the medical disclaimer"""
    return {
        "disclaimer": {
            "title": "Medical Disclaimer",
            "content": """
            IMPORTANT: This AI-powered brain disease detection system is designed for 
            informational and educational purposes only. It is NOT intended to be a 
            substitute for professional medical advice, diagnosis, or treatment.
            
            Key Points:
            1. The AI predictions are based on pattern recognition and may not be 100% accurate.
            2. Always seek the advice of a qualified healthcare provider with any questions 
               about a medical condition.
            3. Never disregard professional medical advice or delay seeking it because of 
               something you have read or seen in this application.
            4. If you think you may have a medical emergency, call your doctor or emergency 
               services immediately.
            5. The information provided by this system does not create a doctor-patient relationship.
            
            By using this system, you acknowledge that you have read and understood this disclaimer.
            """,
            "last_updated": "2024-01-01",
            "accepted_by_usage": True
        }
    }
