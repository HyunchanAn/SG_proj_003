from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="V-SAMS API", description="Vision based Surface Analysis for Material Status")

class RoughnessRequest(BaseModel):
    image_data: str

class RoughnessResponse(BaseModel):
    roughness: float
    gloss: float

@app.post("/analyze/roughness", response_model=RoughnessResponse)
def analyze_roughness(req: RoughnessRequest):
    """
    Mock API for Roughness Analysis
    In a real scenario, this would decode the base64 image and use SurfaceEvaluator
    """
    if not req.image_data:
        raise HTTPException(status_code=400, detail="Image data is empty")
    
    # Returning dummy data for integration tests
    return RoughnessResponse(
        roughness=0.15,
        gloss=100.0
    )
