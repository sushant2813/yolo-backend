from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()
model = YOLO("best.pt")  # Load your trained YOLOv8 model

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    results = model(image)

    # Convert YOLO results to JSON
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    
    return JSONResponse(content={"detections": detections})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
