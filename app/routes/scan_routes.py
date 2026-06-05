"""
Brain Disease AI - Scan Routes
Brain scan upload, AI analysis, and result retrieval APIs
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
import logging
from datetime import datetime

from app.database import SessionLocal, User, BrainScan, ScanStatus, DiseaseType, ScanReport, get_db
from app.schemas import (
    ScanResponse, ScanDetail, ScanReportResponse,
    PredictionHistory, MessageResponse
)
from app.auth.security import get_current_active_user, get_current_user_flexible
from app.services.file_service import file_upload_service
from app.services.email_service import send_scan_result_notification
from app.ai_models.predictor import BrainDiseasePredictor

router = APIRouter(tags=["Brain Scans"])
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# 1. CANONICAL DISEASE MAPPINGS
# ─────────────────────────────────────────
VALID_DISEASE_TYPES = [
    'GLIOMA', 'MENINGIOMA', 'PITUITARY', 'NO_TUMOR',
    'VERY_MILD_DEMENTED', 'MILD_DEMENTED',
    'MODERATE_DEMENTED', 'NON_DEMENTED',
    'NORMAL', 'INCONCLUSIVE', 'STROKE', 'NO_STROKE'
]

_DISEASE_ENUM_MAP: dict = {v.lower(): v for v in VALID_DISEASE_TYPES}

def resolve_disease_enum(label: str) -> "DiseaseType | None":
    if not label:
        return None
    key = label.strip().lower()
    canonical = _DISEASE_ENUM_MAP.get(key)
    if canonical is None:
        logger.warning("resolve_disease_enum: no match for %r", label)
        return None
    try:
        return DiseaseType(canonical)
    except ValueError:
        return None

def map_prediction(disease_key: str) -> str:
    normalised = (
        disease_key.lower()
        .strip()
        .replace(" / ", "/")
        .replace(" ", "_")
        .replace("/", "_")
    )

    mapping = {
        "glioma": "GLIOMA",
        "meningioma": "MENINGIOMA",
        "pituitary": "PITUITARY",
        "no_tumor": "NO_TUMOR",
        "very_mild_demented": "VERY_MILD_DEMENTED",
        "mild_demented": "MILD_DEMENTED",
        "moderate_demented": "MODERATE_DEMENTED",
        "non_demented": "NON_DEMENTED",
        "stroke": "STROKE",
        "no_stroke": "NO_STROKE", 
        "normal": "NORMAL",
        "healthy": "NORMAL",
        "model_offline": "INCONCLUSIVE",
        "inconclusive": "INCONCLUSIVE",
    }
    return mapping.get(normalised, "INCONCLUSIVE")

# ─────────────────────────────────────────
# 2. UPLOAD & PROCESSING
# ─────────────────────────────────────────
@router.post("/upload", response_model=ScanResponse, status_code=status.HTTP_201_CREATED)
async def upload_scan(
    background_tasks: BackgroundTasks,
    scan_type: str = Form(...),
    notes: Optional[str] = Form(None),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    valid_types = ["MRI", "CT", "PET"]
    if scan_type.upper() not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scan type. Must be one of: {', '.join(valid_types)}"
        )
    
    file_path, original_name, file_size, file_hash = await file_upload_service.save_upload(
        file=file,
        user_id=current_user.id
    )
    
    scan = BrainScan(
        user_id=current_user.id,
        scan_type=scan_type.upper(),
        file_path=file_path,
        file_name=original_name,
        file_size=file_size,
        file_hash=file_hash,
        notes=notes,
        status=ScanStatus.PENDING
    )
    
    db.add(scan)
    db.commit()
    db.refresh(scan)
    
    background_tasks.add_task(process_scan_async, scan.id)
    return scan

async def process_scan_async(scan_id: int):
    from app.database import SessionLocal
    from app.ai_models.predictor import BrainDiseasePredictor

    db = SessionLocal()
    scan = None

    try:
        scan = db.query(BrainScan).filter(BrainScan.id == scan_id).first()
        if not scan:
            return

        scan.status = ScanStatus.PROCESSING
        db.commit()

        predictor = BrainDiseasePredictor()
        result = await predictor.predict(scan.file_path, scan.scan_type)

        raw_prediction = result.get("predicted_disease") or "INCONCLUSIVE"
        confidence     = float(result.get("confidence") or 0.0)
        disease_type   = str(result.get("disease_type") or "unknown")

        mapped_key = map_prediction(raw_prediction)
        
        # In UI, we usually want "NO_STROKE" and "NORMAL" to just show as "NORMAL" for consistency
        ui_prediction = "NORMAL" if mapped_key in ["NO_STROKE", "NORMAL"] else mapped_key
        scan.predicted_disease = resolve_disease_enum(ui_prediction)

        # Retrieve per-model breakdown from the predictor results
        model_results = result.get("all_model_results", {})
        t_res = model_results.get("tumor_model", {})
        a_res = model_results.get("alz_model", {})
        s_res = model_results.get("stroke_model", {})

        scan.confidence_score = confidence
        scan.all_predictions  = {
            "final_disease": raw_prediction,
            "final_conf": confidence,
            "disease_type": disease_type,
            "confidence_level": result.get("confidence_level", "ERROR"),
            "tumor_model": t_res,
            "alz_model": a_res,
            "stroke_model": s_res
        }
        
        scan.tumor_result     = t_res.get("label", "NO_TUMOR")
        scan.tumor_confidence = t_res.get("confidence", 0.0)
        scan.alz_result       = a_res.get("label", "NON_DEMENTED")
        scan.alz_confidence   = a_res.get("confidence", 0.0)
        scan.status           = ScanStatus.COMPLETED
        scan.processed_at     = datetime.utcnow()

        report = create_scan_report(scan, result)
        db.add(report)
        db.commit()

        # Email Notification
        try:
            user = db.query(User).filter(User.id == scan.user_id).first()
            if user:
                await send_scan_result_notification(
                    email=user.email,
                    name=user.first_name or user.username,
                    scan_id=scan_id,
                    disease=raw_prediction,
                    confidence=confidence,
                )
        except Exception as email_err:
            logger.warning(f"Scan {scan_id}: email notification failed (non-fatal)")

    except Exception as e:
        logger.error(f"process_scan_async failed for scan_id={scan_id}: {e}", exc_info=True)
        try:
            if scan is not None:
                scan.status = ScanStatus.FAILED
                scan.notes  = f"Processing error: {str(e)[:500]}"
                db.commit()
        except Exception as db_err:
            logger.error(f"Could not update scan status to FAILED: {db_err}")
    finally:
        try:
            db.close()
        except Exception:
            pass

def create_scan_report(scan: BrainScan, result: dict) -> ScanReport:
    try:
        from app.config import TREATMENTS, DISEASES
    except ImportError:
        TREATMENTS, DISEASES = {}, {}

    try:
        predicted  = result.get("predicted_disease") or "INCONCLUSIVE"
        confidence = float(result.get("confidence") or 0.0)
        disease_key = predicted.lower().replace(" ", "_")
        
        disease_info   = DISEASES.get(disease_key, {}) if isinstance(DISEASES, dict) else {}
        treatment_info = TREATMENTS.get(disease_key, {}) if isinstance(TREATMENTS, dict) else {}
        disease_name = disease_info.get("name") or predicted

        summary = f"AI analysis detected {disease_name} with {confidence * 100:.1f}% confidence."
        
        findings = (
            f"Based on the {scan.scan_type} scan analysis, the AI model has identified "
            f"patterns consistent with {disease_name}. "
        )
        
        recommendations = (
            "1. Consult a qualified neurologist for proper diagnosis.\n"
            "2. Consider additional diagnostic tests as recommended by your doctor.\n"
            "3. Review the treatment suggestions provided.\n"
            "4. Follow precautionary measures to manage symptoms."
        )

        suggested = []
        if isinstance(treatment_info.get("medications"), list):
            suggested += treatment_info["medications"]
        if isinstance(treatment_info.get("procedures"), list):
            suggested += treatment_info["procedures"]

        return ScanReport(
            scan_id=scan.id,
            summary=summary,
            detailed_findings=findings,
            recommendations=recommendations,
            suggested_treatments=suggested,
            precautions=treatment_info.get("lifestyle") or [],
            lifestyle_guidance=disease_info.get("risk_factors") or [],
        )

    except Exception as e:
        logger.error(f"create_scan_report failed: {e}", exc_info=True)
        return ScanReport(
            scan_id=scan.id,
            summary="Report generation encountered an error. Please consult a physician.",
            detailed_findings="Automated report generation failed.",
            recommendations="Please consult a qualified neurologist.",
            suggested_treatments=[],
            precautions=[],
            lifestyle_guidance=[],
        )

# ─────────────────────────────────────────
# 3. RETRIEVAL & MANAGEMENT ROUTES
# ─────────────────────────────────────────
@router.get("/", response_model=List[ScanResponse])
async def get_my_scans(
    status_filter: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    query = db.query(BrainScan).filter(BrainScan.user_id == current_user.id)
    
    if status_filter:
        try:
            status_enum = ScanStatus(status_filter.lower())
            query = query.filter(BrainScan.status == status_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid status filter")
    
    return query.order_by(BrainScan.created_at.desc()).offset(skip).limit(limit).all()

@router.get("/history", response_model=PredictionHistory)
async def get_prediction_history(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    total_scans = db.query(func.count(BrainScan.id)).filter(
        BrainScan.user_id == current_user.id
    ).scalar()
    
    predictions = db.query(BrainScan).filter(
        BrainScan.user_id == current_user.id,
        BrainScan.status == ScanStatus.COMPLETED
    ).order_by(BrainScan.processed_at.desc()).all()
    
    return PredictionHistory(total_scans=total_scans, predictions=predictions)

@router.get("/{scan_id}", response_model=ScanDetail)
async def get_scan_detail(
    scan_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    scan = db.query(BrainScan).filter(BrainScan.id == scan_id, BrainScan.user_id == current_user.id).first()
    if not scan: raise HTTPException(status_code=404, detail="Scan not found")
    
    report = db.query(ScanReport).filter(ScanReport.scan_id == scan_id).first()
    
    scan_dict = {
        "id": scan.id,
        "user_id": scan.user_id,
        "scan_type": scan.scan_type,
        "file_name": scan.file_name,
        "file_size": scan.file_size,
        "status": scan.status,
        "predicted_disease": scan.predicted_disease,
        "confidence_score": scan.confidence_score,
        "all_predictions": {
            "tumor_model": {"label": scan.tumor_result, "confidence": scan.tumor_confidence},
            "alz_model": {"label": scan.alz_result, "confidence": scan.alz_confidence}
        },
        "notes": scan.notes,
        "scan_date": scan.scan_date,
        "processed_at": scan.processed_at,
        "created_at": scan.created_at,
        "doctor_review": scan.doctor_review,
        "reviewed_at": scan.reviewed_at,
        "report": report
    }
    return ScanDetail(**scan_dict)

@router.get("/{scan_id}/report", response_model=ScanReportResponse)
async def get_scan_report(
    scan_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    scan = db.query(BrainScan).filter(BrainScan.id == scan_id, BrainScan.user_id == current_user.id).first()
    if not scan: raise HTTPException(status_code=404, detail="Scan not found")
    
    report = db.query(ScanReport).filter(ScanReport.scan_id == scan_id).first()
    if not report: raise HTTPException(status_code=404, detail="Report not available yet")
    
    return report

@router.get("/{scan_id}/image")
async def get_scan_image(
    scan_id: int,
    current_user: User = Depends(get_current_user_flexible),
    db: Session = Depends(get_db)
):
    scan = db.query(BrainScan).filter(BrainScan.id == scan_id, BrainScan.user_id == current_user.id).first()
    if not scan: raise HTTPException(status_code=404, detail="Scan not found")
    
    file_path = file_upload_service.get_file_path(scan.file_path)
    if not file_path: raise HTTPException(status_code=404, detail="Scan file not found")
    
    return FileResponse(path=str(file_path), filename=scan.file_name, media_type="application/octet-stream")

@router.delete("/{scan_id}", response_model=MessageResponse)
async def delete_scan(
    scan_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    scan = db.query(BrainScan).filter(BrainScan.id == scan_id, BrainScan.user_id == current_user.id).first()
    if not scan: raise HTTPException(status_code=404, detail="Scan not found")
    
    await file_upload_service.delete_file(scan.file_path)
    db.query(ScanReport).filter(ScanReport.scan_id == scan_id).delete()
    db.delete(scan)
    db.commit()
    
    return MessageResponse(message="Scan deleted successfully")

@router.post("/{scan_id}/reprocess", response_model=ScanResponse)
async def reprocess_scan(
    scan_id: int,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    scan = db.query(BrainScan).filter(BrainScan.id == scan_id, BrainScan.user_id == current_user.id).first()
    if not scan: raise HTTPException(status_code=404, detail="Scan not found")
    
    if scan.status not in [ScanStatus.FAILED, ScanStatus.PENDING]:
        raise HTTPException(status_code=400, detail="Only failed or pending scans can be reprocessed")
    
    scan.status = ScanStatus.PENDING
    scan.predicted_disease = None
    scan.confidence_score = None
    scan.all_predictions = None
    scan.processed_at = None
    db.commit()
    
    background_tasks.add_task(process_scan_async, scan.id)
    db.refresh(scan)
    return scan