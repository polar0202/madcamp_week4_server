# Style Transfer GPU Server

AdaIN 기반 실시간 스타일 변환 GPU 서버 (하이브리드 아키텍처 Option A용)

## 요구사항

- Python 3.10+
- CUDA 지원 GPU (RTX 3090 권장)
- PyTorch 2.0+

## 설치

```bash
cd server
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1용
python scripts/download_models.py
# 기본 스타일 이미지 추가 (필수)
cp your_style_image.jpg styles/default.jpg
```

모델 다운로드가 실패하면 [릴리스 페이지](https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0)에서 `decoder.pth`, `vgg_normalised.pth`를 수동으로 받아 `models/` 디렉터리에 넣으세요.

## 스타일 설정 (둘 중 하나 필수)

AdaIN 모델은 **스타일이 반드시 필요**합니다.

### 방법 1: 미리 인코딩된 스타일 (권장, FPS 향상)

여러 이미지를 인코딩해 한 번만 계산 후 캐시 사용. 매 요청마다 스타일 VGG 인코딩을 건너뜀.

```bash
# 1) 지브리 이미지 크롤링
python scripts/crawl_ghibli_images.py
# 또는 사람 이미지: python scripts/crawl_ghibli_people.py

# 2) 인코딩하여 ghibli_style.pt 저장
python scripts/encode_styles.py
```

서버 재시작 시 `styles/ghibli_style.pt`를 자동으로 사용합니다.

### 방법 2: 단일 이미지 (default.jpg)

```bash
cp your_style_image.jpg styles/default.jpg
```

`ghibli_style.pt`가 없으면 `default.jpg`를 사용하며, 요청마다 스타일을 VGG로 인코딩합니다.

## 실행

```bash
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 80
```

또는 `./run.sh` (포트 80). root 권한 필요 시 `sudo ./run.sh`

GPU 사용 확인: `http://[VM_IP]/health` (KCLOUD: VPN 연결 후 접속)

## API

### POST /stylize

멀티파트 폼 데이터:
- `content_image`: 필수. 마스크 영역 크롭 이미지 (JPEG/PNG)
- `style_image`: 선택. 스타일 이미지 (기본 스타일 미설정 시 필수)
- `alpha`: 선택. 스타일 강도 0.0~1.0 (기본: 1.0)

응답: JPEG 이미지 바이트

### WebSocket /ws

실시간 스트리밍용. JSON 메시지:

**전송**
- `{"type": "content", "data": "<base64 content image>"}`: 콘텐츠 이미지
- `{"type": "style", "data": "<base64 style image>"}`: 스타일 이미지 (선택)
- `{"type": "stylize", "alpha": 0.8}`: 스타일 변환 실행

**수신**
- `{"type": "result", "data": "<base64 stylized image>"}`: 결과
- `{"type": "error", "message": "..."}`: 오류

## 클라이언트 연동 흐름

1. 모바일에서 MediaPipe로 세그멘테이션 마스크 생성
2. 마스크 바운딩 박스로 영역 크롭
3. 크롭 이미지를 이 서버로 전송 (POST 또는 WebSocket)
4. 스타일 변환 결과 수신
5. 원본 프레임에 마스크 기반으로 합성

## 성능

- RTX 3090 기준: 256x256~384x384 해상도에서 약 30~100 FPS
- 입력 해상도가 클수록 처리 시간 증가
