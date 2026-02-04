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

app = FastAPI(
    title="Ghibli Style Transfer Server",
    description="Ghibli Diffusion style transfer (Img2Img + Canny + LCM + IP-Adapter)",
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
        kwargs = {"content_size": 512}
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
    prompt: str = Form(None),
    negative_prompt: str = Form(None),
    controlnet_scale: float = Form(1.0),
    guidance_scale: float = Form(1.0),
    num_inference_steps: int = Form(6),
    seed: int = Form(42),
):
    """
    Ghibli Diffusion: Canny + LCM + IP-Adapter 고정 사용.
    """
    engine = engine or registry.default_engine_name
    content_bytes = await content_image.read()
    
    try:
        if engine != "ghibli_diffusion":
            raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}")
        
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
        
        return Response(content=result, media_type="image/jpeg")
    
    except Exception as e:
        logger.exception("stylize failed")
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"{str(e)}\n\n{tb}")


@app.post("/stylize_fast")
async def stylize_fast(
    content_image: UploadFile = File(...),
    engine: str | None = Form(None),
    controlnet_scale: float = Form(1.0),
    seed: int = Form(42),
):
    """
    Fast mode. Ghibli: LCM 6 steps + IP-Adapter 고정.
    """
    engine = engine or registry.default_engine_name
    content_bytes = await content_image.read()
    
    try:
        if engine != "ghibli_diffusion":
            raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}")
        
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
    - {"type": "set_engine", "engine": "ghibli_diffusion"}
    - {"type": "content", "data": "<base64 image>"}
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
                engine_name = msg.get("engine", "ghibli_diffusion")
                if engine_name == "ghibli_diffusion":
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
                    if current_engine == "ghibli_diffusion":
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
