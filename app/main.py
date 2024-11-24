from fastapi import FastAPI, File, UploadFile
import time
import os
import shutil 

app = FastAPI()
IMG_DIR = "./upload_img"
os.makedirs(IMG_DIR, exist_ok=True) 

def generate_id():
    timestamp = int(time.time() * 1000)
    return str(timestamp)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/upload")
async def save_img(file: UploadFile = File(...)):

    file_name = generate_id() + ".jpeg"
    file_path = os.path.join(IMG_DIR, file_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"success": True}




