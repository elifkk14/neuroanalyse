# NeuroAnalyse Demo

This package is a runtime-only demo. It includes the web interface, backend, demo MRI files, the dementia mask, and the Late Fusion model checkpoints needed for inference.

It does not include training scripts, training notebooks, training CSV files, preprocessing notebooks, or experiment folders.

## Start

```bash
cd NeuroAnalyse_Demo
cd interface/backend
pip install -r requirements.txt
cd ../..
./start.sh
```

Open:

```text
http://127.0.0.1:8001
```

Demo clinician login:

```text
Username: ayse.yilmaz
Password: Test1234
```

Demo admin login:

```text
Username: admin@neuroanalyse.local
Password: Admin1234
```

## Included Model

The interface uses `LateFusion-v1.0`, combining:

- `model_runtime/full_model.ckpt`
- `model_runtime/masked_model.ckpt`

The final prediction is the 50/50 average of the full-brain and masked-region model outputs.
