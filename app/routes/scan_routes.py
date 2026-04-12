"""
Brain Disease AI - Scan Routes
Brain scan upload, AI analysis, and result retrieval APIs
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
import logging
from datetime import datetime
from app.database import SessionLocal, User, BrainScan, ScanStatus, DiseaseType, ScanReport, get_db
from app.schemas import (
    ScanCreate, ScanResponse, ScanDetail, ScanReportResponse,
    PredictionResult, PredictionHistory, MessageResponse
)
from app.auth.security import get_current_active_user, get_current_user_flexible
from app.services.file_service import file_upload_service
from app.services.email_service import send_scan_result_notification
from app.ai_models import BrainDiseasePredictor

router = APIRouter(tags=["Brain Scans"])

logger = logging.getLogger(__name__)

# Valid DiseaseType enum values – source of truth for enum string lookup
VALID_DISEASE_TYPES = [
    'GLIOMA', 'MENINGIOMA', 'PITUITARY', 'NO_TUMOR',
    'VERY_MILD_DEMENTED', 'MILD_DEMENTED',
    'MODERATE_DEMENTED', 'NON_DEMENTED',
    'NORMAL', 'INCONCLUSIVE'
]

# Case-insensitive enum lookup map: any capitalisation of a key → DiseaseType
# Built once at import time so it's fast and never mis-keyed.
_DISEASE_ENUM_MAP: dict = {v.lower(): v for v in VALID_DISEASE_TYPES}


def resolve_disease_enum(label: str) -> "DiseaseType | None":
    """
    Safely resolve an arbitrary string to a DiseaseType enum member.
    Returns None if no match is found (caller must handle gracefully).
    """
    if not label:
        return None
    key = label.strip().lower()
    canonical = _DISEASE_ENUM_MAP.get(key)
    if canonical is None:
        logger.warning("resolve_disease_enum: no match for %r – defaulting to None", label)
        return None
    try:
        return DiseaseType(canonical)
    except ValueError:
        logger.error("resolve_disease_enum: DiseaseType(%r) raised ValueError", canonical)
        return None


def map_prediction(disease_key: str) -> str:
    """Normalise a raw prediction label to a canonical VALID_DISEASE_TYPES string."""
    normalised = (
        disease_key.lower()
        .strip()
        .replace(" / ", "/")
        .replace(" ", "_")
        .replace("/", "_")
    )

    mapping = {
        # Tumor
        "glioma":       "GLIOMA",
        "meningioma":   "MENINGIOMA",
        "pituitary":    "PITUITARY",
        "no_tumor":     "NO_TUMOR",
        # Alzheimer
        "very_mild_demented": "VERY_MILD_DEMENTED",
        "mild_demented":      "MILD_DEMENTED",
        "moderate_demented":  "MODERATE_DEMENTED",
        "non_demented":       "NON_DEMENTED",
        # Normal / healthy aliases (from decision engine)
        "normal":                           "NORMAL",
        "healthy":                          "NORMAL",
        "healthy_no_significant_findings": "NORMAL",
        "inconclusive":                     "INCONCLUSIVE",
    }

    result = mapping.get(normalised, "INCONCLUSIVE")
    logger.info(
        "map_prediction | raw=%r  normalised=%r  mapped=%s",
        disease_key, normalised, result
    )
    return result


@router.post("/upload", response_model=ScanResponse, status_code=status.HTTP_201_CREATED)
async def upload_scan(
    background_tasks: BackgroundTasks,
    scan_type: str = "MRI",
    notes: Optional[str] = None,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Upload a brain scan image for AI analysis.
    
    - **scan_type**: Type of scan (MRI, CT, PET)
    - **notes**: Optional notes about the scan
    - **file**: Brain scan image file (JPEG, PNG, DICOM, NIfTI)
    
    The scan will be processed by the AI model in the background.
    """
    # Validate scan type
    valid_types = ["MRI", "CT", "PET"]
    if scan_type.upper() not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scan type. Must be one of: {', '.join(valid_types)}"
        )
    
    # Save uploaded file
    file_path, original_name, file_size, file_hash = await file_upload_service.save_upload(
        file=file,
        user_id=current_user.id
    )
    
    # Create scan record
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
    
    # Process scan in background
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
            logger.warning(f"process_scan_async: scan_id={scan_id} not found in DB")
            return

        scan.status = ScanStatus.PROCESSING
        db.commit()

        # ── Run AI prediction (always returns a valid dict) ───────────────
        predictor = BrainDiseasePredictor()
        result = await predictor.predict(scan.file_path, scan.scan_type)

        # ── Validate result structure; fill defaults if anything is missing ─
        raw_prediction = result.get("predicted_disease") or "INCONCLUSIVE"
        confidence     = float(result.get("confidence") or 0.0)
        disease_type   = str(result.get("disease_type") or "unknown")

        all_preds      = result.get("all_predictions") or {}
        tumor_data     = all_preds.get("tumor_model") or {}
        alz_data       = all_preds.get("alz_model")   or {}

        tumor_label    = str(tumor_data.get("label") or "NO_TUMOR")
        tumor_conf     = float(tumor_data.get("confidence") or 0.0)
        alz_label      = str(alz_data.get("label") or "NON_DEMENTED")
        alz_conf       = float(alz_data.get("confidence") or 0.0)

        logger.info(
            "Scan %d | disease_type=%s  final=%s(%.2f)  "
            "tumor=%s(%.2f)  alz=%s(%.2f)",
            scan_id, disease_type, raw_prediction, confidence,
            tumor_label, tumor_conf, alz_label, alz_conf
        )

        # ── Map label → canonical DB enum string ──────────────────────────
        mapped_key = map_prediction(raw_prediction)

        # ── Resolve to DiseaseType enum (safe, never raises) ──────────────
        scan.predicted_disease = resolve_disease_enum(mapped_key)
        if scan.predicted_disease is None:
            logger.warning(
                "Scan %d: could not map %r to DiseaseType – stored as NULL",
                scan_id, mapped_key
            )

        # ── Persist results ───────────────────────────────────────────────
        scan.confidence_score = confidence
        scan.all_predictions  = {
            # top-level decision fields (used by the UI)
            "final_disease":  raw_prediction,
            "final_conf":     confidence,
            "disease_type":   disease_type,
            # per-model raw results
            "tumor_model": {"label": tumor_label, "confidence": tumor_conf},
            "alz_model":   {"label": alz_label,   "confidence": alz_conf},
        }
        scan.tumor_result     = tumor_label
        scan.tumor_confidence = tumor_conf
        scan.alz_result       = alz_label
        scan.alz_confidence   = alz_conf
        scan.status           = ScanStatus.COMPLETED
        scan.processed_at     = datetime.utcnow()

        report = create_scan_report(scan, result)
        db.add(report)
        db.commit()
        logger.info(f"Scan {scan_id} completed and persisted successfully")

        # ── Send email notification (non-blocking – failure must NOT abort) ─
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
            logger.warning(f"Scan {scan_id}: email notification failed (non-fatal): {email_err}")

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
    """Create a detailed scan report – never raises."""
    try:
        from app.config import TREATMENTS, DISEASES

        # Safe key extraction with defaults
        predicted  = result.get("predicted_disease") or "INCONCLUSIVE"
        confidence = float(result.get("confidence") or 0.0)

        disease_key    = predicted.lower().replace(" ", "_")
        disease_info   = DISEASES.get(disease_key, {})   if isinstance(DISEASES, dict)   else {}
        treatment_info = TREATMENTS.get(disease_key, {}) if isinstance(TREATMENTS, dict) else {}

        disease_name = disease_info.get("name") or predicted

        summary = (
            f"AI analysis detected {disease_name} "
            f"with {confidence * 100:.1f}% confidence."
        )
        findings = (
            f"Based on the brain scan analysis, the AI model has identified "
            f"patterns consistent with {disease_name}. "
            f"The model analyzed various features in the scan and compared them "
            f"against trained patterns for MRI tumor and Alzheimer classes."
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
        # Return minimal valid report so the scan record is not blocked
        return ScanReport(
            scan_id=scan.id,
            summary="Report generation encountered an error. Please consult a physician.",
            detailed_findings="Automated report generation failed.",
            recommendations="Please consult a qualified neurologist.",
            suggested_treatments=[],
            precautions=[],
            lifestyle_guidance=[],
        )


@router.get("/", response_model=List[ScanResponse])
async def get_my_scans(
    status_filter: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get list of user's brain scans.
    
    - **status_filter**: Filter by status (pending, processing, completed, failed)
    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum number of records to return
    """
    query = db.query(BrainScan).filter(BrainScan.user_id == current_user.id)
    
    if status_filter:
        try:
            status_enum = ScanStatus(status_filter.lower())
            query = query.filter(BrainScan.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid status filter"
            )
    
    scans = query.order_by(BrainScan.created_at.desc()).offset(skip).limit(limit).all()
    return scans


@router.get("/history", response_model=PredictionHistory)
async def get_prediction_history(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get user's complete prediction history"""
    total_scans = db.query(func.count(BrainScan.id)).filter(
        BrainScan.user_id == current_user.id
    ).scalar()
    
    predictions = db.query(BrainScan).filter(
        BrainScan.user_id == current_user.id,
        BrainScan.status == ScanStatus.COMPLETED
    ).order_by(BrainScan.processed_at.desc()).all()
    
    return PredictionHistory(
        total_scans=total_scans,
        predictions=predictions
    )


@router.get("/{scan_id}", response_model=ScanDetail)
async def get_scan_detail(
    scan_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific scan"""
    scan = db.query(BrainScan).filter(
        BrainScan.id == scan_id,
        BrainScan.user_id == current_user.id
    ).first()
    
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan not found"
        )
    
    # Get report if available
    report = db.query(ScanReport).filter(ScanReport.scan_id == scan_id).first()
    
    # Build response
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
            "tumor_model": {
                "label":      scan.tumor_result,
                "confidence": scan.tumor_confidence
            },
            "alz_model": {
                "label":      scan.alz_result,
                "confidence": scan.alz_confidence
            }
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
    """Get the detailed report for a scan"""
    # Verify scan belongs to user
    scan = db.query(BrainScan).filter(
        BrainScan.id == scan_id,
        BrainScan.user_id == current_user.id
    ).first()
    
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan not found"
        )
    
    report = db.query(ScanReport).filter(ScanReport.scan_id == scan_id).first()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not available yet"
        )
    
    return report


@router.get("/{scan_id}/image")
async def get_scan_image(
    scan_id: int,
    current_user: User = Depends(get_current_user_flexible),  # allows cookie auth for <img src="...">
    db: Session = Depends(get_db)
):
    """Download the scan image file"""
    scan = db.query(BrainScan).filter(
        BrainScan.id == scan_id,
        BrainScan.user_id == current_user.id
    ).first()
    
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan not found"
        )
    
    file_path = file_upload_service.get_file_path(scan.file_path)
    
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan file not found"
        )
    
    return FileResponse(
        path=str(file_path),
        filename=scan.file_name,
        media_type="application/octet-stream"
    )


@router.delete("/{scan_id}", response_model=MessageResponse)
async def delete_scan(
    scan_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a scan and its associated files"""
    scan = db.query(BrainScan).filter(
        BrainScan.id == scan_id,
        BrainScan.user_id == current_user.id
    ).first()
    
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan not found"
        )
    
    # Delete the file
    await file_upload_service.delete_file(scan.file_path)
    
    # Delete report if exists
    db.query(ScanReport).filter(ScanReport.scan_id == scan_id).delete()
    
    # Delete scan record
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
    """Reprocess a failed scan"""
    scan = db.query(BrainScan).filter(
        BrainScan.id == scan_id,
        BrainScan.user_id == current_user.id
    ).first()
    
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scan not found"
        )
    
    if scan.status not in [ScanStatus.FAILED, ScanStatus.PENDING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only failed or pending scans can be reprocessed"
        )
    
    # Reset scan status
    scan.status = ScanStatus.PENDING
    scan.predicted_disease = None
    scan.confidence_score = None
    scan.all_predictions = None
    scan.processed_at = None
    db.commit()
    
    # Reprocess in background
    background_tasks.add_task(process_scan_async, scan.id)
    
    db.refresh(scan)
    return scan
