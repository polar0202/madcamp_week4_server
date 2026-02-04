#!/bin/bash
# Multi-Engine Style Transfer Server
# 기본 엔진: ghibli_diffusion (DEFAULT_ENGINE 환경변수로 변경 가능)
# 예: ./run.sh
# KCLOUD: 포트 80 사용 (22, 80, 443만 방화벽 허용)
# 포트 80은 root 권한 필요할 수 있음 → sudo ./run.sh
cd "$(dirname "$0")"

# 첫 번째 인자로 엔진 지정 가능 (예: ./run.sh controlnet)
if [ -n "$1" ]; then
  export DEFAULT_ENGINE="$1"
fi

uvicorn main:app --host 0.0.0.0 --port 80
