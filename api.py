import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from pydantic import BaseModel

from vsams.analysis.surface_evaluator import SurfaceEvaluator
from vsams.utils.substrate_db import SubstrateDB

app = FastAPI(
    title="V-SAMS Integration API",
    description="V-SAMS (SG_proj_003) API for R.A.D.A.R (SG_proj_004) Integration",
    version="0.1.0",
)

# Initialize resources globally so they stay in memory
evaluator = SurfaceEvaluator()
db = SubstrateDB()


class SubstrateMatchResult(BaseModel):
    product_name: str
    category: str
    similarity: float


class VSAMSAnalysisOutput(BaseModel):
    roughness_ra: float
    gloss_percent: float
    detected_finish: str
    confidence: float
    matching_substrates: List[SubstrateMatchResult]


@app.post("/analyze", response_model=VSAMSAnalysisOutput)
async def analyze_surface(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Analyze surface
    result = evaluator.analyze(image)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    ra = result.get("roughness", 0.0)
    gloss = result.get("gloss", 0.0)
    detected_finish = result.get("predicted_label", "Unknown")

    # DB matching (Top 5)
    # Using find_closest_top_k instead of find_closest to return multiple substrates
    matches_raw = db.find_closest_top_k(ra, gloss, k=5)
    matching_substrates = []

    for match in matches_raw:
        matching_substrates.append(
            SubstrateMatchResult(
                product_name=match["product_name"],
                category=match.get("category", "Unknown"),
                similarity=match.get("similarity", 0.0),
            )
        )

    # Calculate confidence based on top match's similarity (simple heuristic)
    confidence = 0.0
    if matching_substrates:
        confidence = matching_substrates[0].similarity

    return VSAMSAnalysisOutput(
        roughness_ra=ra,
        gloss_percent=gloss,
        detected_finish=detected_finish,
        confidence=confidence,
        matching_substrates=matching_substrates,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
