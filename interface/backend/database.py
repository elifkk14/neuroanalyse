from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer,
    LargeBinary, String, Text, create_engine, event, inspect, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "neuroanalyse.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, _):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ── Models ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    full_name = Column(String(128), nullable=False, default="")
    title = Column(String(64), nullable=False, default="")
    department = Column(String(128), nullable=False, default="")
    role = Column(String(16), nullable=False, default="clinician")  # admin | clinician
    language = Column(String(4), nullable=False, default="tr")      # tr | en
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_active = Column(Boolean, nullable=False, default=True)
    is_locked = Column(Boolean, nullable=False, default=False)
    failed_attempts = Column(Integer, nullable=False, default=0)
    last_failed_at = Column(DateTime, nullable=True)
    last_login_at = Column(DateTime, nullable=True)

    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    analyses_requested = relationship("Analysis", foreign_keys="Analysis.requesting_clinician_id", back_populates="requesting_clinician")
    analyses_created = relationship("Analysis", foreign_keys="Analysis.created_by_id", back_populates="created_by")
    audit_logs = relationship("AuditLog", back_populates="user")


class UserSession(Base):
    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token_hash = Column(String(128), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_active_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    ip_address = Column(String(64), nullable=True)
    is_valid = Column(Boolean, nullable=False, default=True)

    user = relationship("User", back_populates="sessions")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(64), unique=True, nullable=False, index=True)
    patient_name = Column(String(128), nullable=False, default="", index=True)
    birth_year = Column(Integer, nullable=False)
    sex = Column(String(8), nullable=False)   # M | F | other
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    analyses = relationship("Analysis", back_populates="patient", order_by="Analysis.scan_date")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    scan_date = Column(String(16), nullable=False)           # YYYY-MM-DD
    requesting_clinician_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    clinician_note = Column(Text, nullable=True)

    # Model outputs
    centiloid = Column(Float, nullable=True)
    confidence_low = Column(Float, nullable=True)
    confidence_high = Column(Float, nullable=True)
    full_prediction = Column(Float, nullable=True)
    masked_prediction = Column(Float, nullable=True)
    risk_category = Column(String(16), nullable=True)        # negative|borderline|elevated|high

    # Visualisation (base64 PNG strings)
    slice_axial = Column(Text, nullable=True)
    slice_coronal = Column(Text, nullable=True)
    slice_sagittal = Column(Text, nullable=True)
    masked_slice_axial = Column(Text, nullable=True)
    masked_slice_coronal = Column(Text, nullable=True)
    masked_slice_sagittal = Column(Text, nullable=True)
    gradcam_axial = Column(Text, nullable=True)
    gradcam_coronal = Column(Text, nullable=True)
    gradcam_sagittal = Column(Text, nullable=True)

    # Processing metadata
    model_version = Column(String(64), nullable=True, default="LateFusion-v1.0")
    processing_time_s = Column(Float, nullable=True)
    status = Column(String(16), nullable=False, default="pending")   # pending|processing|completed|failed
    error_message = Column(Text, nullable=True)

    # MRI lifecycle
    mri_deleted = Column(Boolean, nullable=False, default=False)
    mri_deleted_at = Column(DateTime, nullable=True)
    mri_deleted_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Report lifecycle
    report_path = Column(String(256), nullable=True)
    report_generated_at = Column(DateTime, nullable=True)
    report_generated_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Ownership
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    patient = relationship("Patient", back_populates="analyses")
    requesting_clinician = relationship("User", foreign_keys=[requesting_clinician_id], back_populates="analyses_requested")
    created_by = relationship("User", foreign_keys=[created_by_id], back_populates="analyses_created")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(64), nullable=False, index=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String(64), nullable=True)

    user = relationship("User", back_populates="audit_logs")


class SystemSetting(Base):
    __tablename__ = "system_settings"

    key = Column(String(64), primary_key=True)
    value = Column(Text, nullable=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_setting(db: Session, key: str, default: str | None = None) -> str | None:
    row = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    return row.value if row else default


def set_setting(db: Session, key: str, value: str) -> None:
    row = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if row:
        row.value = value
    else:
        db.add(SystemSetting(key=key, value=value))
    db.commit()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _run_lightweight_migrations()


def _run_lightweight_migrations() -> None:
    inspector = inspect(engine)
    if "patients" not in inspector.get_table_names():
        return
    patient_cols = {col["name"] for col in inspector.get_columns("patients")}
    with engine.begin() as conn:
        if "patient_name" not in patient_cols:
            conn.execute(text("ALTER TABLE patients ADD COLUMN patient_name VARCHAR(128) NOT NULL DEFAULT ''"))
        if "analyses" in inspector.get_table_names():
            analysis_cols = {col["name"] for col in inspector.get_columns("analyses")}
            for col_name in ("masked_slice_axial", "masked_slice_coronal", "masked_slice_sagittal"):
                if col_name not in analysis_cols:
                    conn.execute(text(f"ALTER TABLE analyses ADD COLUMN {col_name} TEXT"))
