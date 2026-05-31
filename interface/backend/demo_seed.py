"""
Demo Data Seeder
================
3 hasta / 5 analiz — gerçek ADNI NIfTI dosyalarından üretilmiş.

TC numaraları algoritmik olarak üretilmiş, gerçek kişilere ait değildir.

Hasta 1 — Serkan Aydın (46281573994) — 3 tarama, progresyon vakası
  2018-04-15 : -3.3  CL  → Negatif
  2021-09-08 : 15.5  CL  → Negatif (yükselen)
  2024-03-22 : 52.6  CL  → Yüksek

Hasta 2 — Elif Kaya (53742961894) — 1 tarama
  2023-11-14 : 10.7  CL  → Negatif

Hasta 3 — Mustafa Şen (71538642994) — 1 tarama
  2024-01-30 : 56.2  CL  → Yüksek
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEMOS_DIR    = _PROJECT_ROOT / "mri_demos_nii"

# ── Gerçek inference sonuçları (LateFusionPredictor çıktısı) ──────────────────
_DEMO_DATA = [
    {
        "patient_id":   "46281573994",
        "patient_name": "Serkan Aydın",
        "birth_year":   1958,
        "sex":          "M",
        "analyses": [
            {
                "scan_date":   "2018-04-15",
                "centiloid":   -3.3,
                "full_pred":   -23.5,
                "masked_pred": 16.9,
                "nii_file":    "wr_I173792_023_S_0376_T1(1).nii",
                "note":        "Rutin kontrol taraması.",
                "proc_time":   134.8,
            },
            {
                "scan_date":   "2021-09-08",
                "centiloid":   15.5,
                "full_pred":   5.1,
                "masked_pred": 25.9,
                "nii_file":    "wr_I180310_002_S_2010_T1.nii",
                "note":        "Takip taraması. Değerlerde hafif artış.",
                "proc_time":   128.4,
            },
            {
                "scan_date":   "2024-03-22",
                "centiloid":   52.6,
                "full_pred":   71.8,
                "masked_pred": 33.4,
                "nii_file":    "wr_I174005_016_S_1326_T1.nii",
                "note":        "Belirgin artış gözlemlendi. Nöroloji konsültasyonu önerildi.",
                "proc_time":   141.2,
            },
        ],
    },
    {
        "patient_id":   "53742961894",
        "patient_name": "Elif Kaya",
        "birth_year":   1965,
        "sex":          "F",
        "analyses": [
            {
                "scan_date":   "2023-11-14",
                "centiloid":   10.7,
                "full_pred":   -5.2,
                "masked_pred": 26.5,
                "nii_file":    "wr_I179549_128_S_2002_T1.nii",
                "note":        None,
                "proc_time":   122.6,
            },
        ],
    },
    {
        "patient_id":   "71538642994",
        "patient_name": "Mustafa Şen",
        "birth_year":   1952,
        "sex":          "M",
        "analyses": [
            {
                "scan_date":   "2024-01-30",
                "centiloid":   56.2,
                "full_pred":   90.1,
                "masked_pred": 22.2,
                "nii_file":    "wr_I180722_016_S_1117_T1.nii",
                "note":        "Nöroloji konsültasyonuna yönlendirildi.",
                "proc_time":   139.7,
            },
        ],
    },
]

MVP_ADMIN_EMAIL    = "admin@neuroanalyse.local"
MVP_ADMIN_PASSWORD = "Admin1234"


def _generate_slices_from_nii(nii_path: str) -> dict[str, str | None]:
    """NIfTI dosyasından raw + masked slice PNG'leri üretir (base64)."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from pipeline import _preprocess_nifti, generate_mri_slices

        with tempfile.TemporaryDirectory(prefix="na_seed_") as tmp:
            full_path, masked_path = _preprocess_nifti(nii_path, tmp)
            raw    = generate_mri_slices(full_path)
            masked = generate_mri_slices(masked_path)

        return {
            "slice_axial":          raw.get("axial"),
            "slice_coronal":        raw.get("coronal"),
            "slice_sagittal":       raw.get("sagittal"),
            "masked_slice_axial":   masked.get("axial"),
            "masked_slice_coronal": masked.get("coronal"),
            "masked_slice_sagittal":masked.get("sagittal"),
        }
    except Exception as exc:
        print(f"[seed] Slice üretilemedi ({nii_path}): {exc}")
        return {}


def seed_demo_data(db) -> None:
    """Veritabanı boşsa demo verileri oluşturur."""
    from database import Analysis, Patient, User
    from auth import hash_password
    from pipeline import classify_centiloid

    _ensure_mvp_admin(db, User, hash_password)

    if db.query(Patient).count() > 0:
        return

    print("[seed] Demo hastalar oluşturuluyor…")

    admin = db.query(User).filter(User.username == MVP_ADMIN_EMAIL).first()

    clinician = db.query(User).filter(User.username == "ayse.yilmaz").first()
    if clinician is None:
        clinician = User(
            username="ayse.yilmaz",
            password_hash=hash_password("Test1234"),
            full_name="Dr. Ayşe Yılmaz",
            title="Uzm. Dr.",
            department="Nöroloji",
            role="clinician",
            language="tr",
        )
        db.add(clinician)
    db.flush()

    for pd in _DEMO_DATA:
        patient = Patient(
            patient_id=pd["patient_id"],
            patient_name=pd["patient_name"],
            birth_year=pd["birth_year"],
            sex=pd["sex"],
            created_by_id=admin.id,
        )
        db.add(patient)
        db.flush()

        for ad in pd["analyses"]:
            cl      = ad["centiloid"]
            risk    = classify_centiloid(cl)
            spread  = abs(ad["full_pred"] - ad["masked_pred"])
            ci_half = max(10.0, spread * 1.2)

            nii_path = str(_DEMOS_DIR / ad["nii_file"])
            slices: dict = {}
            if Path(nii_path).exists():
                print(f"[seed] Slice üretiliyor: {ad['nii_file']} …")
                slices = _generate_slices_from_nii(nii_path)
            else:
                print(f"[seed] Uyarı: {nii_path} bulunamadı, slice atlandı.")

            analysis = Analysis(
                patient_id=patient.id,
                scan_date=ad["scan_date"],
                requesting_clinician_id=clinician.id,
                clinician_note=ad.get("note"),
                centiloid=cl,
                full_prediction=ad["full_pred"],
                masked_prediction=ad["masked_pred"],
                confidence_low=round(cl - ci_half, 1),
                confidence_high=round(cl + ci_half, 1),
                risk_category=risk["risk_category"],
                model_version="LateFusion-v1.0",
                processing_time_s=ad["proc_time"],
                status="completed",
                mri_deleted=True,
                mri_deleted_at=datetime.utcnow(),
                mri_deleted_by_id=admin.id,
                slice_axial=slices.get("slice_axial"),
                slice_coronal=slices.get("slice_coronal"),
                slice_sagittal=slices.get("slice_sagittal"),
                masked_slice_axial=slices.get("masked_slice_axial"),
                masked_slice_coronal=slices.get("masked_slice_coronal"),
                masked_slice_sagittal=slices.get("masked_slice_sagittal"),
                created_by_id=admin.id,
                created_at=datetime.strptime(ad["scan_date"], "%Y-%m-%d"),
            )
            db.add(analysis)

    db.commit()
    print("[seed] Demo veriler oluşturuldu.")


def ensure_demo_assets(db) -> None:
    """Tamamlanan demo analizler için eksik raporları üretir."""
    from database import Analysis, Patient, REPORTS_DIR, User, get_setting
    from pipeline import classify_centiloid
    from report_gen import generate_pdf

    demo_ids = {p["patient_id"] for p in _DEMO_DATA}
    completed = (
        db.query(Analysis)
        .join(Patient, Patient.id == Analysis.patient_id)
        .filter(Analysis.status == "completed", Patient.patient_id.in_(demo_ids))
        .order_by(Analysis.id)
        .all()
    )

    changed = False
    for analysis in completed:
        report_exists = bool(analysis.report_path and Path(analysis.report_path).exists())
        if report_exists:
            continue

        patient  = db.query(Patient).filter(Patient.id == analysis.patient_id).first()
        clinician = db.query(User).filter(User.id == analysis.requesting_clinician_id).first()
        if not patient or analysis.centiloid is None:
            continue

        prev = (
            db.query(Analysis)
            .filter(
                Analysis.patient_id == analysis.patient_id,
                Analysis.id != analysis.id,
                Analysis.status == "completed",
            )
            .order_by(Analysis.scan_date)
            .all()
        )
        prev_data = [
            {
                "scan_date":     p.scan_date,
                "centiloid":     p.centiloid,
                "risk_category": p.risk_category,
                "risk_label":    p.risk_category,
            }
            for p in prev
        ]

        risk      = classify_centiloid(analysis.centiloid)
        report_id = f"NA-DEMO-{analysis.id:06d}"
        pdf_bytes = generate_pdf(
            report_id=report_id,
            patient_id=patient.patient_id,
            patient_name=patient.patient_name,
            birth_year=patient.birth_year,
            sex=patient.sex,
            scan_date=analysis.scan_date,
            analysis_date=analysis.created_at.strftime("%Y-%m-%d"),
            requesting_clinician=clinician.full_name if clinician else "—",
            clinician_note=analysis.clinician_note,
            centiloid=analysis.centiloid,
            confidence_low=analysis.confidence_low or 0.0,
            confidence_high=analysis.confidence_high or 0.0,
            risk_category=analysis.risk_category or "negative",
            risk_label=risk["risk_label_tr"],
            interpretation=risk["interpretation_tr"],
            institution_name=get_setting(db, "institution_name") or "NeuroAnalyse",
            institution_logo_b64=get_setting(db, "institution_logo"),
            slice_axial=analysis.slice_axial,
            slice_coronal=analysis.slice_coronal,
            slice_sagittal=analysis.slice_sagittal,
            previous_analyses=prev_data if prev_data else None,
            is_demo=True,
            language="tr",
        )
        report_path = REPORTS_DIR / f"{report_id}.pdf"
        report_path.write_bytes(pdf_bytes)
        analysis.report_path = str(report_path)
        analysis.report_generated_at = datetime.utcnow()
        analysis.report_generated_by_id = analysis.created_by_id
        changed = True

    if changed:
        db.commit()


def _ensure_mvp_admin(db, User, hash_password) -> None:
    admin = db.query(User).filter(User.username == MVP_ADMIN_EMAIL).first()
    if admin:
        admin.role         = "admin"
        admin.is_active    = True
        admin.is_locked    = False
        admin.failed_attempts = 0
        db.commit()
        return

    db.add(User(
        username=MVP_ADMIN_EMAIL,
        password_hash=hash_password(MVP_ADMIN_PASSWORD),
        full_name="Sistem Yöneticisi",
        title="Dr.",
        department="Nöroloji",
        role="admin",
        language="tr",
    ))
    db.commit()
