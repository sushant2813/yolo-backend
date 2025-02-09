import os  # For environment variable access
import uvicorn  # Make sure uvicorn is imported
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load the trained YOLOv8 model (replace 'yolo11n.pt' with your model file)
model = YOLO("yolo11n.pt")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Open the uploaded file
    image = Image.open(io.BytesIO(await file.read()))
    
    # Run inference on the image
    results = model(image)
    
    # Convert YOLO results to JSON
    detections = results.pandas().xyxy[0].to_dict(orient="records")
    
    return JSONResponse(content={"detections": detections})

# Only required for local testing, remove for production deployment on Render
if __name__ == '__main__':
    # Set the port dynamically from environment variables or default to 5000
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)  # Run with dynamic port
