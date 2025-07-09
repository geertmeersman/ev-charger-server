#!/bin/sh
uvicorn app.main:app --host 0.0.0.0 --port ${WEB_PORT:-8000} --log-level info
