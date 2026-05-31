# NeuroAnalyse Demo

NeuroAnalyse is a web-based demo application that estimates amyloid burden on the Centiloid scale from 3D T1-weighted MRI scans.

This repository is prepared as a runtime-only demo package. It does not include training scripts, training notebooks, training CSV files, preprocessing experiments, or model development folders.

## What's Included

- FastAPI backend
- Single-file React frontend
- Demo patients and sample MRI scans
- PDF report generation
- Late Fusion inference runtime
- Selected full-brain and masked-region model checkpoints

## Model

The interface uses `LateFusion-v1.0`.

The model combines two separate 3D ResNet predictions:

- `model_runtime/full_model.ckpt`
- `model_runtime/masked_model.ckpt`

The final Centiloid estimate is computed as a 50/50 average of the full-brain and masked-region model outputs.

## Setup

Python 3.11 is recommended.

```bash
git clone https://github.com/elifkk14/neuroanalyse.git
cd neuroanalyse
cd interface/backend
pip install -r requirements.txt
cd ../..
./start.sh
```

After the server starts, open:

```text
http://127.0.0.1:8001
```

## Demo Login

Clinician account:

```text
Username: ayse.yilmaz
Password: Test1234
```

Admin account:

```text
Username: admin@neuroanalyse.local
Password: Admin1234
```

## Notes

- The demo database and sample reports are created automatically on first launch.
- Uploaded MRI files are not retained after analysis.
- This demo is intended for research and product demonstration purposes only. It is not approved for clinical diagnosis.

