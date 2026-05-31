#!/bin/bash
cd "$(dirname "$0")/interface/backend"
python3.11 -m uvicorn main:app --host 127.0.0.1 --port 8001 --reload

