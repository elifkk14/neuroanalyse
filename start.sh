#!/bin/bash
cd "$(dirname "$0")/interface/backend"
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload
