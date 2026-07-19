from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import base64
import numpy as np
import cv2
import sys
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent path to import vsams
sys.path.append(str(Path(__file__).parent.parent))
from vsams.analysis.surface_evaluator import SurfaceEvaluator

app = FastAPI(title="V-SAMS API", description="Vision based Surface Analysis for Material Status")
evaluator = SurfaceEvaluator()

import torch
import gc
from fastapi import Request

@app.middleware("http")
async def clear_vram_middleware(request: Request, call_next):
    response = await call_next(request)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return response

from shared_schemas.p003_vsams import RoughnessRequest, RoughnessResponse

@app.post("/analyze/roughness", response_model=RoughnessResponse)
def analyze_roughness(req: RoughnessRequest):
    if not req.image_data:
        raise HTTPException(status_code=400, detail="Image data is empty")
    
    img = None
    try:
        # Decode base64 image if valid
        if req.image_data.startswith("data:image"):
            header, encoded = req.image_data.split(",", 1)
        else:
            encoded = req.image_data
            
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Base64 decode failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format or decode failed")
        
    if img is None:
        logger.error("Decoded image is empty or invalid")
        raise HTTPException(status_code=400, detail="Decoded image is empty or invalid")

    # Convert BGR to RGB and run real SurfaceEvaluator
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    result = evaluator.analyze(pil_img)
    
    return RoughnessResponse(
        roughness=float(result["roughness"]),
        gloss=float(result["gloss"])
    )

