from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import numpy as np
import cv2
import pickle
import os

app = FastAPI()

@app.post("/pred")
async def upload_file(image: UploadFile):
    classes = ['Ariel Sharon','Colin Powell', 'Donald Rumsfeld', 'George W Bush',
    'Gerhard Schroeder', 'Hugo Chavez', 'Junichiro Koizumi', 'Tony Blair']
    
    with open('model01.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    try:
        with open(f"temp_{image.filename}", "wb") as temp_file:
           shutil.copyfileobj(image.file, temp_file)
        
        img = cv2.imread(f"temp_{image.filename}",cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (47, 62))
        img = np.array(img)
        img_1d = img.flatten()
        img_2d = np.array([img_1d],dtype=np.float32)
        min_value = img_2d.min()
        max_value = img_2d.max()
        img_rescaled = (img_2d - min_value) / (max_value - min_value)
        pred = model.predict(img_rescaled)
        pred_name=classes[pred[0]]
        
        
        return JSONResponse(content={"message": "Image uploaded successfully", "Prediction":pred_name})#,img_array
    except Exception as e:
        return JSONResponse(content={"message": "An error occurred", "error": str(e)}, status_code=500)

@app.on_event("startup")
def startup_event():
    for filename in os.listdir():
        if filename.startswith("temp_"):
            os.remove(filename)