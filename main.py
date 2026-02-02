"""GPU Style Transfer Server - Multi-Engine FastAPI."""

import json
import logging
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect, HTTPException, Query

logger = logging.getLogger(__name__)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from engine_registry import registry

# Configuration
MODELS_DIR = Path(__file__).parent / "models"
STYLES_DIR = Path(__file__).parent / "styles"

app = FastAPI(
    title="Multi-Engine Style Transfer Server",
    description="Support multiple style transfer engines (AdaIN, Ghibli Diffusion, etc.)",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    engines = registry.list_engines()
    return {
        "status": "ok",
        "message": "Multi-Engine Style Transfer Server",
        "default_engine": registry.default_engine_name,
        "available_engines": engines,
        "endpoints": {
            "stylize": "POST /stylize - Apply style transfer with engine selection",
            "engines": "GET /engines - List all available engines",
            "health": "GET /health - Check server status",
            "ws": "WebSocket /ws - Real-time streaming",
        },
    }


@app.get("/engines")
async def list_engines():
    """List all available engines with metadata."""
    return {
        "engines": registry.list_engines(),
        "default": registry.default_engine_name,
    }


@app.get("/health")
async def health():
    try:
        kwargs = {"models_dir": MODELS_DIR} if registry.default_engine_name == "adain" else {"content_size": 512}
        default_engine = registry.get_default_engine(**kwargs)
        return {
            "status": "ok",
            "gpu": default_engine.device == "cuda",
            "device": default_engine.device,
            "default_engine": registry.default_engine_name,
            "loaded_engines": [name for name, info in registry.list_engines().items() if info["loaded"]],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/stylize")
async def stylize(
    content_image: UploadFile = File(...),
    engine: str | None = Form(None),
    # AdaIN parameters
    style_image: UploadFile | None = File(None),
    alpha: float = Form(1.0),
    # Ghibli Diffusion (Canny + LCM + IP-Adapter 고정 사용)
    prompt: str = Form(None),
    negative_prompt: str = Form(None),
    controlnet_scale: float = Form(1.0),
    guidance_scale: float = Form(1.0),
    num_inference_steps: int = Form(6),
    seed: int = Form(42),
):
    """
    Engine: 'adain' or 'ghibli_diffusion'
    Ghibli: Canny + LCM + IP-Adapter 항상 사용 (IP-Adapter 옵션은 서버 고정)
    """
    engine = engine or registry.default_engine_name
    content_bytes = await content_image.read()
    
    try:
        # Get engine with appropriate config
        if engine == "adain":
            engine_instance = registry.get_engine(
                "adain",
                models_dir=MODELS_DIR,
                content_size=384,
                style_size=384,
                alpha=alpha,
            )
            
            # Load default style if exists
            if not hasattr(engine_instance, '_default_style_loaded'):
                default_style_path = STYLES_DIR / "default.jpg"
                if default_style_path.exists():
                    with open(default_style_path, "rb") as f:
                        engine_instance.set_default_style(f.read())
                engine_instance._default_style_loaded = True
            
            style_bytes = None
            if style_image:
                style_bytes = await style_image.read()
            
            result = engine_instance.stylize(
                content_image=content_bytes,
                style_image=style_bytes,
                alpha=alpha,
            )
        
        elif engine == "ghibli_diffusion":
            engine_instance = registry.get_engine(
                "ghibli_diffusion",
                content_size=512,
                controlnet_scale=controlnet_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                default_seed=seed,
            )
            result = engine_instance.stylize(
                content_image=content_bytes,
                prompt=prompt,
                negative_prompt=negative_prompt,
                controlnet_scale=controlnet_scale,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}")
        
        return Response(content=result, media_type="image/jpeg")
    
    except Exception as e:
        logger.exception("stylize failed")
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\n{tb}")


@app.post("/stylize_fast")
async def stylize_fast(
    content_image: UploadFile = File(...),
    engine: str | None = Form(None),
    alpha: float = Form(1.0),
    controlnet_scale: float = Form(1.0),
    seed: int = Form(42),
):
    """
    Fast mode. Ghibli: LCM 6 steps + IP-Adapter 항상 사용 (옵션 고정).
    """
    engine = engine or registry.default_engine_name
    content_bytes = await content_image.read()
    
    try:
        if engine == "adain":
            engine_instance = registry.get_engine("adain", models_dir=MODELS_DIR)
            result = engine_instance.stylize(
                content_image=content_bytes,
                alpha=alpha,
            )
        
        elif engine == "ghibli_diffusion":
            engine_instance = registry.get_engine(
                "ghibli_diffusion",
                num_inference_steps=6,
            )
            result = engine_instance.stylize(
                content_image=content_bytes,
                controlnet_scale=controlnet_scale,
                num_inference_steps=6,
                seed=seed,
            )
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}")
        
        return Response(content=result, media_type="image/jpeg")
    
    except Exception as e:
        logger.exception("stylize_fast failed")
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\n{tb}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket protocol for real-time style transfer.
    
    Send (JSON):
    - {"type": "set_engine", "engine": "adain|ghibli_diffusion"}
    - {"type": "content", "data": "<base64 image>"}
    - {"type": "style", "data": "<base64 image>"} (AdaIN only)
    - {"type": "stylize", ...params...}
    
    Receive (JSON):
    - {"type": "result", "data": "<base64 stylized image>"}
    - {"type": "ack", ...}
    - {"type": "error", "message": "..."}
    """
    import base64

    await websocket.accept()

    current_engine = registry.default_engine_name
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

            if msg_type == "set_engine":
                engine_name = msg.get("engine", "adain")
                if engine_name in ["adain", "ghibli_diffusion"]:
                    current_engine = engine_name
                    await websocket.send_json({
                        "type": "ack",
                        "field": "engine",
                        "engine": current_engine
                    })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown engine: {engine_name}"
                    })

            elif msg_type == "content":
                data = msg.get("data")
                if data:
                    pending_content = base64.b64decode(data)
                    await websocket.send_json({"type": "ack", "field": "content"})

            elif msg_type == "style":
                data = msg.get("data")
                if data:
                    pending_style = base64.b64decode(data)
                    await websocket.send_json({"type": "ack", "field": "style"})

            elif msg_type == "stylize":
                if pending_content is None:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No content image received"
                    })
                    continue

                try:
                    if current_engine == "adain":
                        engine_instance = registry.get_engine("adain", models_dir=MODELS_DIR)
                        alpha = msg.get("alpha", 1.0)
                        result = engine_instance.stylize(
                            content_image=pending_content,
                            style_image=pending_style,
                            alpha=alpha,
                        )
                    
                    elif current_engine == "ghibli_diffusion":
                        engine_instance = registry.get_engine(
                            "ghibli_diffusion",
                            content_size=512,
                        )
                        result = engine_instance.stylize(
                            content_image=pending_content,
                            prompt=msg.get("prompt"),
                            negative_prompt=msg.get("negative_prompt"),
                            controlnet_scale=msg.get("controlnet_scale", 1.0),
                            guidance_scale=msg.get("guidance_scale", 1.0),
                            num_inference_steps=msg.get("steps", 6),
                            seed=msg.get("seed", 42),
                        )
                    
                    else:
                        raise ValueError(f"Unknown engine: {current_engine}")
                    
                    result_b64 = base64.b64encode(result).decode("utf-8")
                    await websocket.send_json({
                        "type": "result",
                        "data": result_b64,
                        "engine": current_engine
                    })
                
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
