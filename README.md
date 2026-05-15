# NeuroAnalyse — MRI-Based Amyloid Burden Prediction

A clinical-grade web system that estimates amyloid burden from standard 3D T1-weighted MRI scans — without PET imaging.

---

## What It Does

NeuroAnalyse takes a preprocessed structural MRI (NIfTI format), runs it through a Late Fusion deep learning model, and returns:

- **Centiloid score** — a standardized measure of amyloid burden
- **Risk category** — Negative / Borderline / Elevated / High
- **Clinical interpretation** — plain-language summary for the clinician
- **PDF report** — downloadable, archivable report with MRI slices and score history

---

## Key Features

- **Late Fusion architecture** — two 3D ResNet models (full brain + masked region) fused at prediction time for improved accuracy
- **Longitudinal tracking** — multiple scans per patient visualized as a time series
- **Role-based access** — Admin and Clinician roles with separate permissions
- **Session management** — secure cookie-based sessions with configurable timeout
- **Audit logging** — all actions are logged
- **Demo mode** — pre-loaded sample patients with real ADNI-derived scan results

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, SQLAlchemy, SQLite |
| ML | PyTorch Lightning, 3D ResNet |
| Frontend | React (via CDN, no build step), single HTML file |
| Reports | ReportLab PDF generation |
| Serving | Uvicorn |

---

## Project Structure

```
neuroanalyse/
├── interface/
│   ├── backend/
│   │   ├── main.py              # FastAPI app, all routes
│   │   ├── pipeline.py          # MRI preprocessing + inference
│   │   ├── report_gen.py        # PDF report generation
│   │   ├── auth.py              # Auth, sessions, roles
│   │   ├── database.py          # SQLAlchemy models
│   │   ├── license_manager.py   # License / demo quota
│   │   └── demo_seed.py         # Demo patient data seeder
│   └── frontend/
│       └── NeuroAnalyse.html    # Single-file React frontend
├── Late_Fusion_SON/             # Model training code
│   ├── Train3D_Full_SON/        # Full-brain model
│   └── Train3D_Masked_SON/      # Masked-region model
├── checkpoints/                 # Trained model weights (.ckpt)
├── DementiaMask_AAL3.nii        # Brain mask (AAL3 atlas)
├── mrı_demos_nii/               # Demo NIfTI scans (ADNI-derived)
└── start.sh                     # Server startup script
```

---

## Getting Started

**Requirements:** Python 3.11, pip

```bash
# Install dependencies
cd interface/backend
pip install -r requirements.txt

# Start the server
cd ../..
./start.sh
```

Then open `http://localhost:8001` in your browser.

**Demo login:**
```
Username: ayse.yilmaz
Password: Test1234
```

---

## Demo Patients

Three pre-loaded patients with real inference results (ADNI data):

| Patient | Scans | Centiloid Range | Notes |
|---|---|---|---|
| Serkan Aydın | 3 | −3.3 → 52.6 CL | Progression case |
| Elif Kaya | 1 | 10.7 CL | Negative |
| Mustafa Şen | 1 | 56.2 CL | High |

---

## Model

The Late Fusion model combines two independently trained 3D ResNet networks:

- **Full model** — trained on the complete brain volume
- **Masked model** — trained on amyloid-relevant regions (AAL3 atlas mask)

Predictions are fused at inference time. Output is mapped to the Centiloid scale.

Checkpoint files are included in `checkpoints/`.

---

## License

Academic / research use only. Not approved for clinical diagnosis.
