#!/bin/bash
# KCLOUD: 포트 80 사용 (22, 80, 443만 방화벽 허용)
# 포트 80은 root 권한 필요할 수 있음 → sudo ./run.sh
cd "$(dirname "$0")"
uvicorn main:app --host 0.0.0.0 --port 80
