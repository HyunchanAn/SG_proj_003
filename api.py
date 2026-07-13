from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from PIL import Image
import sys
from pathlib import Path

# Add project root path to sys.path
sys.path.append(str(Path(__file__).parent))
from vsams.analysis.surface_evaluator import SurfaceEvaluator

app = FastAPI(title="V-SAMS Headless API", version="0.1.0")
evaluator = SurfaceEvaluator()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze/roughness")
async def analyze_roughness(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image file")

        # Convert BGR to RGB and load into PIL Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Run real evaluation
        result = evaluator.analyze(pil_img)

        return JSONResponse(
            content={
                "status": "success",
                "roughness": float(result["roughness"]),
                "gloss": float(result["gloss"])
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8003, reload=True)

