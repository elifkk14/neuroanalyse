"""
NeuroAnalyse Backend — FastAPI Application
==========================================
All routes:
  Auth        POST /api/auth/login, /api/auth/logout, GET /api/auth/me
  Users       GET/POST /api/users, GET/PUT/DELETE /api/users/{id}, POST unlock/reset-password
  Patients    GET /api/patients, GET /api/patients/{pid}, POST /api/patients
  Analyses    POST /api/analyses (upload+pipeline), GET /api/analyses/{id},
              GET /api/analyses/{id}/stream (SSE), DELETE /api/analyses/{id}/mri
  Reports     POST /api/reports/{id}, GET /api/reports/{id}/pdf
  Dashboard   GET /api/dashboard
  Settings    GET/PUT /api/settings
  Audit       GET /api/audit, GET /api/audit/export
  License     GET /api/license, POST /api/license/activate
"""
from __future__ import annotations

import logging
import warnings

warnings.filterwarnings("ignore", message="urllib3.*doesn't match a supported version")
warnings.filterwarnings("ignore", message="Coercing Subquery object")
logging.getLogger("passlib").setLevel(logging.ERROR)

import asyncio
import csv
import io
import json
import os
import secrets
import tempfile
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import (
    BackgroundTasks, Depends, File, HTTPException, Query, Request,
    Response, UploadFile, status,
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth import (
    audit, create_session, get_client_ip, get_current_user,
    hash_password, invalidate_session, require_admin,
    validate_password_strength, verify_password, MAX_FAILED_ATTEMPTS,
)
from database import (
    Analysis, AuditLog, Patient, SystemSetting, User,
    get_db, get_setting, init_db, set_setting, REPORTS_DIR,
)
from license_manager import (
    activate_license, check_can_analyze, get_license_info,
    init_default_license,
)
from pipeline import PipelineStep, classify_centiloid, run_pipeline
from report_gen import generate_pdf

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
_MAX_UPLOAD_BYTES = 2 * 1024 ** 3  # 2 GB
_SUPPORTED_EXTS = (".nii.gz", ".nii", ".dcm", ".zip")

# In-memory progress store: analysis_id → {"step": str, "pct": int, "done": bool, "error": str}
_progress: dict[int, dict] = {}
_progress_lock = threading.Lock()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    _bootstrap()
    yield


def _bootstrap():
    from database import SessionLocal, UserSession
    from demo_seed import ensure_demo_assets, seed_demo_data
    db = SessionLocal()
    try:
        init_default_license(db)
        seed_demo_data(db)
        ensure_demo_assets(db)
        # Invalidate all sessions on startup so no stale admin cookies survive a restart
        db.query(UserSession).update({"is_valid": False})
        db.commit()
    finally:
        db.close()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="NeuroAnalyse", docs_url=None, redoc_url=None, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")


@app.get("/")
def index():
    idx = _FRONTEND / "NeuroAnalyse.html"
    if idx.exists():
        return FileResponse(str(idx))
    return {"status": "ok", "message": "NeuroAnalyse API running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class CreateUserRequest(BaseModel):
    username: str
    password: str
    full_name: str = ""
    title: str = ""
    department: str = ""
    role: str = "clinician"
    language: str = "tr"

class UpdateUserRequest(BaseModel):
    full_name: str | None = None
    title: str | None = None
    department: str | None = None
    role: str | None = None
    language: str | None = None

class UpdateProfileRequest(BaseModel):
    full_name: str | None = None
    title: str | None = None
    department: str | None = None
    language: str | None = None

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class ResetPasswordRequest(BaseModel):
    new_password: str

class CreatePatientRequest(BaseModel):
    patient_id: str
    patient_name: str
    birth_year: int
    sex: str

class CreateAnalysisMetaRequest(BaseModel):
    patient_id: str
    scan_date: str
    requesting_clinician_id: int
    clinician_note: str | None = None

class UpdateSettingsRequest(BaseModel):
    institution_name: str | None = None
    session_timeout_minutes: int | None = None
    institution_logo: str | None = None   # base64 data-url

class ActivateLicenseRequest(BaseModel):
    key: str
    institution_name: str

class GenerateReportRequest(BaseModel):
    language: str = "tr"


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.post("/api/auth/login")
def login(body: LoginRequest, request: Request, db: Session = Depends(get_db)):
    ip = get_client_ip(request)
    user = db.query(User).filter(User.username == body.username.lower()).first()

    if not user or not user.is_active:
        audit(db, "login_failed", f"unknown user: {body.username}", ip_address=ip)
        raise HTTPException(status_code=401, detail="invalid_credentials")

    if user.is_locked:
        audit(db, "login_blocked", f"account locked: {body.username}", ip_address=ip)
        raise HTTPException(status_code=403, detail="account_locked")

    if not verify_password(body.password, user.password_hash):
        user.failed_attempts += 1
        user.last_failed_at = datetime.utcnow()
        if user.failed_attempts >= MAX_FAILED_ATTEMPTS:
            user.is_locked = True
            db.commit()
            audit(db, "account_locked", f"user: {user.username}", user_id=user.id, ip_address=ip)
            raise HTTPException(status_code=403, detail="account_locked")
        db.commit()
        raise HTTPException(status_code=401, detail="invalid_credentials")

    user.failed_attempts = 0
    user.last_login_at = datetime.utcnow()
    db.commit()

    timeout = int(get_setting(db, "session_timeout_minutes") or "720")
    token = create_session(db, user, ip, timeout)
    audit(db, "login_success", None, user_id=user.id, ip_address=ip)

    resp = JSONResponse({"ok": True, "role": user.role, "language": user.language})
    resp.set_cookie("na_session", token, httponly=True, samesite="lax", max_age=timeout * 60)
    resp.headers["X-Session-Token"] = token
    return resp


@app.post("/api/auth/logout")
def logout(request: Request, db: Session = Depends(get_db)):
    from auth import _extract_token
    token = _extract_token(request)
    if token:
        invalidate_session(db, token)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("na_session")
    return resp


@app.get("/api/auth/me")
def me(user: User = Depends(get_current_user)):
    return _user_dict(user)


# ── Users ─────────────────────────────────────────────────────────────────────

@app.get("/api/users")
def list_users(db: Session = Depends(get_db), _: User = Depends(require_admin)):
    users = db.query(User).filter(User.is_active == True).all()
    return [_user_dict(u) for u in users]


@app.get("/api/clinicians")
def list_clinicians(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    users = db.query(User).filter(User.is_active == True).all()
    return [{"id": u.id, "full_name": u.full_name, "role": u.role} for u in users]


@app.post("/api/users", status_code=201)
def create_user(
    body: CreateUserRequest,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    if db.query(User).filter(User.username == body.username.lower()).first():
        raise HTTPException(status_code=409, detail="username_taken")
    err = validate_password_strength(body.password)
    if err:
        raise HTTPException(status_code=422, detail=err)
    u = User(
        username=body.username.lower(),
        password_hash=hash_password(body.password),
        full_name=body.full_name,
        title=body.title,
        department=body.department,
        role=body.role,
        language=body.language,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    audit(db, "user_created", f"username={u.username}", user_id=admin.id,
          ip_address=get_client_ip(request))
    return _user_dict(u)


@app.put("/api/users/{uid}")
def update_user(
    uid: int,
    body: UpdateUserRequest,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    u = _get_user_or_404(db, uid)
    if body.full_name is not None: u.full_name = body.full_name
    if body.title is not None:     u.title = body.title
    if body.department is not None: u.department = body.department
    if body.role is not None:      u.role = body.role
    if body.language is not None:  u.language = body.language
    db.commit()
    audit(db, "user_updated", f"uid={uid}", user_id=admin.id, ip_address=get_client_ip(request))
    return _user_dict(u)


@app.delete("/api/users/{uid}")
def delete_user(
    uid: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    if uid == admin.id:
        raise HTTPException(status_code=400, detail="cannot_delete_self")
    u = _get_user_or_404(db, uid)
    u.is_active = False
    db.commit()
    audit(db, "user_deleted", f"uid={uid} username={u.username}", user_id=admin.id,
          ip_address=get_client_ip(request))
    return {"ok": True}


@app.post("/api/users/{uid}/unlock")
def unlock_user(
    uid: int,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    u = _get_user_or_404(db, uid)
    u.is_locked = False
    u.failed_attempts = 0
    db.commit()
    audit(db, "user_unlocked", f"uid={uid}", user_id=admin.id, ip_address=get_client_ip(request))
    return {"ok": True}


@app.post("/api/users/{uid}/reset-password")
def reset_password(
    uid: int,
    body: ResetPasswordRequest,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    err = validate_password_strength(body.new_password)
    if err:
        raise HTTPException(status_code=422, detail=err)
    u = _get_user_or_404(db, uid)
    u.password_hash = hash_password(body.new_password)
    db.commit()
    audit(db, "password_reset", f"uid={uid}", user_id=admin.id, ip_address=get_client_ip(request))
    return {"ok": True}


# ── Profile (self) ────────────────────────────────────────────────────────────

@app.put("/api/profile")
def update_profile(
    body: UpdateProfileRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if body.full_name is not None: user.full_name = body.full_name
    if body.title is not None:     user.title = body.title
    if body.department is not None: user.department = body.department
    if body.language is not None:  user.language = body.language
    db.commit()
    return _user_dict(user)


@app.post("/api/profile/change-password")
def change_password(
    body: ChangePasswordRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(status_code=401, detail="wrong_current_password")
    err = validate_password_strength(body.new_password)
    if err:
        raise HTTPException(status_code=422, detail=err)
    user.password_hash = hash_password(body.new_password)
    db.commit()
    return {"ok": True}


# ── Patients ──────────────────────────────────────────────────────────────────

@app.get("/api/patients")
def list_patients(
    search: str = Query(default=""),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    q = db.query(Patient)
    if search:
        q = q.filter(
            (Patient.patient_id.ilike(f"%{search}%")) |
            (Patient.patient_name.ilike(f"%{search}%"))
        )
    patients = q.order_by(Patient.patient_id).all()
    return [_patient_summary(p, db) for p in patients]


@app.get("/api/patients/{patient_id}")
def get_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    p = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="patient_not_found")
    analyses = _analyses_list(p.analyses)
    return {
        **_patient_summary(p, db),
        "analyses": analyses,
    }


@app.delete("/api/patients/{patient_id}", status_code=204)
def delete_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    p = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="patient_not_found")
    for a in p.analyses:
        db.delete(a)
    db.delete(p)
    db.commit()
    audit(db, "patient_deleted", f"patient_id={patient_id}", user.id)
    return Response(status_code=204)


@app.post("/api/patients", status_code=201)
def create_patient(
    body: CreatePatientRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    body.patient_id = _normalize_tc(body.patient_id)
    body.patient_name = body.patient_name.strip()
    if not body.patient_name:
        raise HTTPException(status_code=422, detail="patient_name_required")
    if not _is_valid_tc(body.patient_id):
        raise HTTPException(status_code=422, detail="invalid_tc")
    if db.query(Patient).filter(Patient.patient_id == body.patient_id).first():
        raise HTTPException(status_code=409, detail="patient_id_exists")
    p = Patient(
        patient_id=body.patient_id,
        patient_name=body.patient_name,
        birth_year=body.birth_year,
        sex=body.sex,
        created_by_id=user.id,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return _patient_summary(p, db)


@app.get("/api/patients/{patient_id}/check")
def check_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    patient_id = _normalize_tc(patient_id)
    if not _is_valid_tc(patient_id):
        raise HTTPException(status_code=422, detail="invalid_tc")
    p = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not p:
        return {"found": False}
    return {
        "found": True,
        "patient_id": p.patient_id,
        "patient_name": p.patient_name,
        "birth_year": p.birth_year,
        "sex": p.sex,
    }


@app.get("/api/patients/search/by-name")
def search_patient_by_name(
    name: str = Query(..., min_length=2),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    q = (
        db.query(Patient)
        .filter(Patient.patient_name.ilike(f"%{name.strip()}%"))
        .order_by(Patient.patient_name)
        .limit(8)
        .all()
    )
    return [_patient_summary(p, db) for p in q]


# ── Analyses ──────────────────────────────────────────────────────────────────

@app.post("/api/analyses", status_code=202)
async def create_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    patient_id: str = Query(...),
    scan_date: str = Query(...),
    requesting_clinician_id: int = Query(...),
    clinician_note: str = Query(default=""),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # License check
    allowed, err = check_can_analyze(db)
    if not allowed:
        raise HTTPException(status_code=402, detail=err)

    filename = file.filename or "upload.bin"
    if not _is_supported(filename):
        raise HTTPException(status_code=400, detail="unsupported_format")

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file_too_large")

    # Get/create patient
    patient_id = _normalize_tc(patient_id)
    if not _is_valid_tc(patient_id):
        raise HTTPException(status_code=422, detail="invalid_tc")
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="patient_not_found")

    # Create analysis record
    analysis = Analysis(
        patient_id=patient.id,
        scan_date=scan_date,
        requesting_clinician_id=requesting_clinician_id,
        clinician_note=clinician_note or None,
        status="processing",
        created_by_id=user.id,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    analysis_id = analysis.id

    with _progress_lock:
        _progress[analysis_id] = {"step": "convert", "pct": 0, "done": False, "error": None}

    # Write upload to temp file (must survive the background task)
    tmp_dir = tempfile.mkdtemp(prefix="na_upload_")
    upload_path = os.path.join(tmp_dir, filename)
    with open(upload_path, "wb") as fh:
        fh.write(content)

    ip = get_client_ip(request)
    audit(db, "analysis_started", f"analysis_id={analysis_id} patient={patient_id}",
          user_id=user.id, ip_address=ip)

    background_tasks.add_task(
        _run_analysis_task,
        analysis_id=analysis_id,
        upload_path=upload_path,
        filename=filename,
        user_id=user.id,
    )

    return {"analysis_id": analysis_id}


def _run_analysis_task(analysis_id: int, upload_path: str, filename: str, user_id: int):
    from database import SessionLocal
    import shutil

    db = SessionLocal()
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        if not analysis:
            return

        def progress_cb(step: str, pct: int):
            with _progress_lock:
                _progress[analysis_id] = {"step": step, "pct": pct, "done": False, "error": None}

        result = run_pipeline(upload_path, filename, progress_cb)

        # Update analysis record
        for key, val in result.items():
            if hasattr(analysis, key):
                setattr(analysis, key, val)

        analysis.status = "completed"
        analysis.mri_deleted = True
        analysis.mri_deleted_at = datetime.utcnow()
        analysis.mri_deleted_by_id = user_id
        db.commit()

        with _progress_lock:
            _progress[analysis_id] = {"step": "done", "pct": 100, "done": True, "error": None}

        from auth import audit as _audit
        _audit(db, "analysis_completed", f"analysis_id={analysis_id} CL={result['centiloid']}",
               user_id=user_id)
        _audit(db, "mri_deleted", f"analysis_id={analysis_id}",
               user_id=user_id)

    except Exception as exc:
        err_str = str(exc)
        if db.query(Analysis).filter(Analysis.id == analysis_id).first():
            analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
            analysis.status = "failed"
            analysis.error_message = err_str[:1000]
            db.commit()
        with _progress_lock:
            _progress[analysis_id] = {"step": "error", "pct": 0, "done": True, "error": err_str}
    finally:
        db.close()
        # Clean up upload file
        try:
            shutil.rmtree(os.path.dirname(upload_path), ignore_errors=True)
        except Exception:
            pass


@app.get("/api/analyses/{aid}/stream")
async def stream_progress(
    aid: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """SSE endpoint for real-time pipeline progress."""
    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            if await request.is_disconnected():
                break
            with _progress_lock:
                state = _progress.get(aid)
            if state:
                yield f"data: {json.dumps(state)}\n\n"
                if state["done"]:
                    break
            else:
                # Analysis not started yet or already cleaned up
                analysis = db.query(Analysis).filter(Analysis.id == aid).first()
                if analysis and analysis.status in ("completed", "failed"):
                    yield f"data: {json.dumps({'step': analysis.status, 'pct': 100, 'done': True, 'error': analysis.error_message})}\n\n"
                    break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/analyses/{aid}")
def get_analysis(
    aid: int,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    a = db.query(Analysis).filter(Analysis.id == aid).first()
    if not a:
        raise HTTPException(status_code=404, detail="not_found")
    return _analysis_full(a, db)


@app.get("/api/analyses")
def list_analyses(
    patient_id: str | None = Query(default=None),
    risk_category: str | None = Query(default=None),
    clinician_id: int | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    q = db.query(Analysis).filter(Analysis.status == "completed")

    if patient_id:
        p = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if p:
            q = q.filter(Analysis.patient_id == p.id)

    if risk_category:
        q = q.filter(Analysis.risk_category == risk_category)
    if clinician_id:
        q = q.filter(Analysis.requesting_clinician_id == clinician_id)
    if date_from:
        q = q.filter(Analysis.scan_date >= date_from)
    if date_to:
        q = q.filter(Analysis.scan_date <= date_to)

    total = q.count()
    analyses = q.order_by(Analysis.scan_date.desc()).offset(offset).limit(limit).all()
    return {"total": total, "items": [_analysis_list_item(a, db) for a in analyses]}


@app.delete("/api/analyses/{aid}/mri")
def delete_mri(
    aid: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    a = db.query(Analysis).filter(Analysis.id == aid).first()
    if not a:
        raise HTTPException(status_code=404, detail="not_found")
    a.mri_deleted = True
    a.mri_deleted_at = datetime.utcnow()
    a.mri_deleted_by_id = user.id
    db.commit()
    audit(db, "mri_deleted", f"analysis_id={aid}", user_id=user.id,
          ip_address=get_client_ip(request))
    return {"ok": True}


# ── Reports ───────────────────────────────────────────────────────────────────

@app.post("/api/reports/{aid}")
def generate_report(
    aid: int,
    body: GenerateReportRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    a = db.query(Analysis).filter(Analysis.id == aid).first()
    if not a:
        raise HTTPException(status_code=404, detail="not_found")
    if a.status != "completed":
        raise HTTPException(status_code=400, detail="analysis_not_ready")

    patient = db.query(Patient).filter(Patient.id == a.patient_id).first()
    clinician = db.query(User).filter(User.id == a.requesting_clinician_id).first()

    # Previous analyses for time series
    prev = (
        db.query(Analysis)
        .filter(
            Analysis.patient_id == a.patient_id,
            Analysis.id != a.id,
            Analysis.status == "completed",
        )
        .order_by(Analysis.scan_date)
        .all()
    )
    prev_data = [
        {
            "scan_date": p.scan_date,
            "centiloid": p.centiloid,
            "risk_category": p.risk_category,
            "risk_label": p.risk_category,
        }
        for p in prev
    ]

    lic = get_license_info(db)
    is_demo = lic.get("is_demo", False)
    institution_name = lic.get("institution_name") or "NeuroAnalyse"
    institution_logo = get_setting(db, "institution_logo")

    lang = body.language
    risk = classify_centiloid(a.centiloid)
    report_id = f"NA-{a.id:06d}-{secrets.token_hex(3).upper()}"

    if a.report_path and Path(a.report_path).exists():
        existing = Path(a.report_path)
        return {"ok": True, "report_id": existing.stem, "filename": existing.name, "immutable": True}

    pdf_bytes = generate_pdf(
        report_id=report_id,
        patient_id=patient.patient_id if patient else "—",
        patient_name=patient.patient_name if patient else None,
        birth_year=patient.birth_year if patient else 0,
        sex=patient.sex if patient else "—",
        scan_date=a.scan_date,
        analysis_date=a.created_at.strftime("%Y-%m-%d"),
        requesting_clinician=clinician.full_name if clinician else "—",
        clinician_note=a.clinician_note,
        centiloid=a.centiloid,
        confidence_low=a.confidence_low or 0.0,
        confidence_high=a.confidence_high or 0.0,
        risk_category=a.risk_category or "negative",
        risk_label=risk[f"risk_label_{lang}"],
        interpretation=risk[f"interpretation_{lang}"],
        model_version=a.model_version or "LateFusion-v1.0",
        processing_time_s=a.processing_time_s or 0.0,
        institution_name=institution_name,
        institution_logo_b64=institution_logo,
        slice_axial=a.slice_axial,
        slice_coronal=a.slice_coronal,
        slice_sagittal=a.slice_sagittal,
        previous_analyses=prev_data if prev else None,
        is_demo=is_demo,
        language=lang,
    )

    # Persist to disk
    report_filename = f"{report_id}.pdf"
    report_path = REPORTS_DIR / report_filename
    report_path.write_bytes(pdf_bytes)

    a.report_path = str(report_path)
    a.report_generated_at = datetime.utcnow()
    a.report_generated_by_id = user.id
    db.commit()

    audit(db, "report_generated", f"analysis_id={aid} report={report_id}",
          user_id=user.id, ip_address=get_client_ip(request))

    return {"ok": True, "report_id": report_id, "filename": report_filename}


@app.get("/api/reports/{aid}/pdf")
def download_report(
    aid: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    a = db.query(Analysis).filter(Analysis.id == aid).first()
    if not a:
        raise HTTPException(status_code=404, detail="not_found")
    if not a.report_path or not Path(a.report_path).exists():
        raise HTTPException(status_code=404, detail="report_not_generated")

    audit(db, "report_downloaded", f"analysis_id={aid}", user_id=user.id,
          ip_address=get_client_ip(request))

    return FileResponse(
        path=a.report_path,
        media_type="application/pdf",
        filename=Path(a.report_path).name,
    )


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.get("/api/dashboard")
def dashboard(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    from sqlalchemy import func
    from datetime import date

    total = db.query(Analysis).filter(Analysis.status == "completed").count()

    first_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    this_month = (
        db.query(Analysis)
        .filter(Analysis.status == "completed", Analysis.created_at >= first_of_month)
        .count()
    )

    # Critical follow-up: CL > 50 and no analysis in last 12 months
    from sqlalchemy import text
    twelve_months_ago = datetime.utcnow().replace(
        year=datetime.utcnow().year - 1
    ).strftime("%Y-%m-%d")
    critical_subq = (
        db.query(Analysis.patient_id)
        .filter(Analysis.centiloid > 50, Analysis.status == "completed")
        .filter(Analysis.scan_date < twelve_months_ago)
        .distinct()
        .scalar_subquery()
    )
    has_recent_subq = (
        db.query(Analysis.patient_id)
        .filter(Analysis.centiloid > 50, Analysis.status == "completed")
        .filter(Analysis.scan_date >= twelve_months_ago)
        .distinct()
        .scalar_subquery()
    )
    critical_count = (
        db.query(Patient.id)
        .filter(Patient.id.in_(critical_subq))
        .filter(Patient.id.notin_(has_recent_subq))
        .count()
    )

    recent = (
        db.query(Analysis)
        .filter(Analysis.status == "completed")
        .order_by(Analysis.created_at.desc())
        .limit(5)
        .all()
    )

    return {
        "total_analyses": total,
        "this_month": this_month,
        "critical_follow_up": critical_count,
        "recent_analyses": [_analysis_list_item(a, db) for a in recent],
    }


# ── Settings ──────────────────────────────────────────────────────────────────

@app.get("/api/settings")
def get_settings(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    result = {
        "institution_name": get_setting(db, "institution_name") or "",
        "session_timeout_minutes": int(get_setting(db, "session_timeout_minutes") or "720"),
        "institution_logo": get_setting(db, "institution_logo"),
        "model_name": "Small3DResNetRegressor (Late Fusion)",
        "model_version": "LateFusion-v1.0",
        "cl_thresholds": {"negative": 25, "borderline": 50, "high": 100},
    }
    if user.role == "admin":
        lic = get_license_info(db)
        result["license"] = lic
    return result


@app.put("/api/settings")
def update_settings(
    body: UpdateSettingsRequest,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    if body.institution_name is not None:
        set_setting(db, "institution_name", body.institution_name)
    if body.session_timeout_minutes is not None:
        t = max(5, min(720, body.session_timeout_minutes))
        set_setting(db, "session_timeout_minutes", str(t))
    if body.institution_logo is not None:
        set_setting(db, "institution_logo", body.institution_logo)
    audit(db, "settings_updated", None, user_id=admin.id, ip_address=get_client_ip(request))
    return {"ok": True}


# ── Audit log ─────────────────────────────────────────────────────────────────

@app.get("/api/audit")
def get_audit(
    action: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0),
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    q = db.query(AuditLog).order_by(AuditLog.timestamp.desc())
    if action:
        q = q.filter(AuditLog.action.ilike(f"%{action}%"))
    if date_from:
        q = q.filter(AuditLog.timestamp >= date_from)
    if date_to:
        q = q.filter(AuditLog.timestamp <= date_to + " 23:59:59")
    total = q.count()
    logs = q.offset(offset).limit(limit).all()
    return {
        "total": total,
        "items": [
            {
                "id": l.id,
                "timestamp": l.timestamp.isoformat(),
                "action": l.action,
                "details": l.details,
                "ip_address": l.ip_address,
                "user_id": l.user_id,
                "username": l.user.username if l.user else None,
            }
            for l in logs
        ],
    }


@app.get("/api/audit/export")
def export_audit(
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).all()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "timestamp", "action", "details", "username", "ip_address"])
    for l in logs:
        writer.writerow([
            l.id,
            l.timestamp.isoformat(),
            l.action,
            l.details or "",
            l.user.username if l.user else "",
            l.ip_address or "",
        ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit_log.csv"},
    )


# ── Reports list ──────────────────────────────────────────────────────────────

@app.get("/api/reports")
def list_reports(
    patient_id: str | None = Query(default=None),
    risk_category: str | None = Query(default=None),
    clinician_id: int | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    limit: int = Query(default=50, le=200),
    offset: int = Query(default=0),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    q = (
        db.query(Analysis)
        .filter(Analysis.status == "completed")
        .filter(Analysis.report_path.isnot(None))
    )
    if patient_id:
        p = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if p:
            q = q.filter(Analysis.patient_id == p.id)
    if risk_category:
        q = q.filter(Analysis.risk_category == risk_category)
    if clinician_id:
        q = q.filter(Analysis.requesting_clinician_id == clinician_id)
    if date_from:
        q = q.filter(Analysis.scan_date >= date_from)
    if date_to:
        q = q.filter(Analysis.scan_date <= date_to)

    total = q.count()
    analyses = q.order_by(Analysis.scan_date.desc()).offset(offset).limit(limit).all()
    return {"total": total, "items": [_analysis_list_item(a, db) for a in analyses]}


@app.get("/api/reports/export-csv")
def export_reports_csv(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    analyses = (
        db.query(Analysis)
        .filter(Analysis.status == "completed")
        .order_by(Analysis.scan_date.desc())
        .all()
    )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "analysis_id", "patient_id", "scan_date", "centiloid",
        "confidence_low", "confidence_high", "risk_category",
        "model_version", "processing_time_s", "report_generated_at",
    ])
    for a in analyses:
        patient = db.query(Patient).filter(Patient.id == a.patient_id).first()
        writer.writerow([
            a.id,
            patient.patient_id if patient else "",
            a.scan_date,
            a.centiloid,
            a.confidence_low,
            a.confidence_high,
            a.risk_category,
            a.model_version,
            a.processing_time_s,
            a.report_generated_at.isoformat() if a.report_generated_at else "",
        ])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=analyses_export.csv"},
    )


# ── License ───────────────────────────────────────────────────────────────────

@app.get("/api/license")
def get_license(db: Session = Depends(get_db), _: User = Depends(require_admin)):
    return get_license_info(db)


@app.post("/api/license/activate")
def activate_lic(
    body: ActivateLicenseRequest,
    request: Request,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin),
):
    result = activate_license(db, body.key, body.institution_name)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "activation_failed"))
    audit(db, "license_activated", f"type={result.get('type')}", user_id=admin.id,
          ip_address=get_client_ip(request))
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_supported(filename: str) -> bool:
    name = filename.lower()
    return any(name.endswith(ext) for ext in _SUPPORTED_EXTS)


def _normalize_tc(value: str) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


def _is_valid_tc(value: str) -> bool:
    tc = _normalize_tc(value)
    if len(tc) != 11 or tc[0] == "0":
        return False
    digits = [int(ch) for ch in tc]
    odd_sum = sum(digits[0:9:2])
    even_sum = sum(digits[1:8:2])
    digit_10 = ((odd_sum * 7) - even_sum) % 10
    digit_11 = sum(digits[:10]) % 10
    return digits[9] == digit_10 and digits[10] == digit_11


def _get_user_or_404(db: Session, uid: int) -> User:
    u = db.query(User).filter(User.id == uid, User.is_active == True).first()
    if not u:
        raise HTTPException(status_code=404, detail="user_not_found")
    return u


def _user_dict(u: User) -> dict:
    return {
        "id": u.id,
        "username": u.username,
        "full_name": u.full_name,
        "title": u.title,
        "department": u.department,
        "role": u.role,
        "language": u.language,
        "is_locked": u.is_locked,
        "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
    }


def _patient_summary(p: Patient, db: Session) -> dict:
    analyses = [a for a in p.analyses if a.status == "completed"]
    last = analyses[-1] if analyses else None
    last_clinician = None
    if last and last.requesting_clinician_id:
        u = db.query(User).filter(User.id == last.requesting_clinician_id).first()
        last_clinician = u.full_name if u else None
    return {
        "id": p.id,
        "patient_id": p.patient_id,
        "patient_name": p.patient_name,
        "birth_year": p.birth_year,
        "sex": p.sex,
        "analysis_count": len(analyses),
        "last_centiloid": last.centiloid if last else None,
        "last_risk_category": last.risk_category if last else None,
        "last_scan_date": last.scan_date if last else None,
        "last_clinician": last_clinician,
    }


def _analysis_full(a: Analysis, db: Session) -> dict:
    patient = db.query(Patient).filter(Patient.id == a.patient_id).first()
    clinician = db.query(User).filter(User.id == a.requesting_clinician_id).first()
    risk = classify_centiloid(a.centiloid) if a.centiloid is not None else {}
    return {
        "id": a.id,
        "patient_id": patient.patient_id if patient else None,
        "patient_name": patient.patient_name if patient else None,
        "birth_year": patient.birth_year if patient else None,
        "sex": patient.sex if patient else None,
        "scan_date": a.scan_date,
        "requesting_clinician": clinician.full_name if clinician else None,
        "clinician_note": a.clinician_note,
        "centiloid": a.centiloid,
        "confidence_low": a.confidence_low,
        "confidence_high": a.confidence_high,
        "full_prediction": a.full_prediction,
        "masked_prediction": a.masked_prediction,
        "risk_category": a.risk_category,
        "risk_label_tr": risk.get("risk_label_tr"),
        "risk_label_en": risk.get("risk_label_en"),
        "interpretation_tr": risk.get("interpretation_tr"),
        "interpretation_en": risk.get("interpretation_en"),
        "slice_axial": a.slice_axial,
        "slice_coronal": a.slice_coronal,
        "slice_sagittal": a.slice_sagittal,
        "masked_slice_axial": a.masked_slice_axial,
        "masked_slice_coronal": a.masked_slice_coronal,
        "masked_slice_sagittal": a.masked_slice_sagittal,
        "model_version": a.model_version,
        "processing_time_s": a.processing_time_s,
        "status": a.status,
        "error_message": a.error_message,
        "mri_deleted": a.mri_deleted,
        "report_ready": bool(a.report_path and Path(a.report_path).exists()),
        "created_at": a.created_at.isoformat(),
    }


def _analysis_list_item(a: Analysis, db: Session) -> dict:
    patient = db.query(Patient).filter(Patient.id == a.patient_id).first()
    clinician = db.query(User).filter(User.id == a.requesting_clinician_id).first()
    risk = classify_centiloid(a.centiloid) if a.centiloid is not None else {}
    return {
        "id": a.id,
        "patient_id": patient.patient_id if patient else None,
        "patient_name": patient.patient_name if patient else None,
        "birth_year": patient.birth_year if patient else None,
        "scan_date": a.scan_date,
        "centiloid": a.centiloid,
        "risk_category": a.risk_category,
        "risk_label_tr": risk.get("risk_label_tr"),
        "risk_label_en": risk.get("risk_label_en"),
        "requesting_clinician": clinician.full_name if clinician else None,
        "report_ready": bool(a.report_path and Path(a.report_path).exists()),
        "created_at": a.created_at.isoformat(),
    }


def _analyses_list(analyses) -> list:
    from database import SessionLocal
    db = SessionLocal()
    try:
        items = []
        for a in analyses:
            if a.status != "completed":
                continue
            clinician = db.query(User).filter(User.id == a.requesting_clinician_id).first()
            risk = classify_centiloid(a.centiloid) if a.centiloid is not None else {}
            items.append({
                "id": a.id,
                "status": "completed",
                "scan_date": a.scan_date,
                "centiloid": a.centiloid,
                "confidence_low": a.confidence_low,
                "confidence_high": a.confidence_high,
                "risk_category": a.risk_category,
                "risk_label_tr": risk.get("risk_label_tr"),
                "risk_label_en": risk.get("risk_label_en"),
                "interpretation_tr": risk.get("interpretation_tr"),
                "interpretation_en": risk.get("interpretation_en"),
                "clinician": clinician.full_name if clinician else None,
                "clinician_note": a.clinician_note,
                "slice_axial": a.slice_axial,
                "slice_coronal": a.slice_coronal,
                "slice_sagittal": a.slice_sagittal,
                "model_version": a.model_version,
                "processing_time_s": a.processing_time_s,
                "report_ready": bool(a.report_path and Path(a.report_path).exists()),
                "mri_deleted": a.mri_deleted,
            })
        return items
    finally:
        db.close()
