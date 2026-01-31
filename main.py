"""GPU Style Transfer Server - FastAPI + WebSocket."""

import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from style_transfer import StyleTransferEngine

# Configuration
MODELS_DIR = Path(__file__).parent / "models"
STYLES_DIR = Path(__file__).parent / "styles"
DEFAULT_CONTENT_SIZE = 384
DEFAULT_STYLE_SIZE = 384

app = FastAPI(
    title="Style Transfer GPU Server",
    description="AdaIN-based arbitrary style transfer for hybrid camera filter",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine - lazy init
_engine: StyleTransferEngine | None = None


def get_engine() -> StyleTransferEngine:
    global _engine
    if _engine is None:
        _engine = StyleTransferEngine(
            models_dir=MODELS_DIR,
            content_size=DEFAULT_CONTENT_SIZE,
            style_size=DEFAULT_STYLE_SIZE,
            alpha=1.0,
            preserve_color=False,
        )
        _engine.load_models()
        # Load default style if available
        default_style_path = STYLES_DIR / "default.jpg"
        if default_style_path.exists():
            with open(default_style_path, "rb") as f:
                _engine.set_default_style(f.read())
    return _engine


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Style Transfer GPU Server",
        "endpoints": {
            "stylize": "POST /stylize - multipart: content_image, style_image (optional)",
            "ws": "WebSocket /ws - binary protocol for real-time",
            "health": "GET /health",
        },
    }


@app.get("/health")
async def health():
    try:
        engine = get_engine()
        return {"status": "ok", "gpu": engine.device == "cuda"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/stylize")
async def stylize(
    content_image: UploadFile = File(...),
    style_image: UploadFile | None = File(None),
    alpha: float = 1.0,
):
    """
    Apply style transfer to content image.
    - content_image: Required. Cropped region from client (JPEG/PNG)
    - style_image: Optional. Uses server default if not provided
    - alpha: Style strength 0.0-1.0 (default: 1.0)
    """
    engine = get_engine()

    content_bytes = await content_image.read()
    style_bytes = None
    if style_image:
        style_bytes = await style_image.read()

    try:
        result = engine.stylize(
            content_image=content_bytes,
            style_image=style_bytes,
            alpha=alpha,
        )
        from fastapi.responses import Response

        return Response(content=result, media_type="image/jpeg")
    except ValueError as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket protocol for real-time style transfer:
    - Send: JSON message with type and data
      - {"type": "content", "data": "<base64 image>"} - content image
      - {"type": "style", "data": "<base64 image>"} - style image (optional per request)
      - {"type": "stylize", "alpha": 0.8} - trigger stylize with optional alpha
    - Receive: {"type": "result", "data": "<base64 stylized image>"}
    """
    import base64

    await websocket.accept()

    engine = get_engine()
    pending_content: bytes | None = None
    pending_style: bytes | None = None

    try:
        while True:
            raw = await websocket.receive()
            text = raw.get("text")
            if not text:
                continue

            try:
                msg = json.loads(text)
                msg_type = msg.get("type")
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            if msg_type == "content":
                data = msg.get("data")
                if data:
                    pending_content = base64.b64decode(data)
                    await websocket.send_json({"type": "ack", "field": "content"})

            elif msg_type == "style":
                data = msg.get("data")
                if data:
                    pending_style = base64.b64decode(data)
                    engine.set_default_style(pending_style)
                    await websocket.send_json({"type": "ack", "field": "style"})

            elif msg_type == "stylize":
                if pending_content is None:
                    await websocket.send_json(
                        {"type": "error", "message": "No content image received"}
                    )
                    continue

                alpha = msg.get("alpha", 1.0)
                try:
                    result = engine.stylize(
                        content_image=pending_content,
                        style_image=pending_style,
                        alpha=alpha,
                    )
                    result_b64 = base64.b64encode(result).decode("utf-8")
                    await websocket.send_json(
                        {"type": "result", "data": result_b64}
                    )
                except Exception as e:
                    await websocket.send_json(
                        {"type": "error", "message": str(e)}
                    )

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
