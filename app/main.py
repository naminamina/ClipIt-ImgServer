import os
import time
import shutil 
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
IMG_DIR = "./upload_img"
os.makedirs(IMG_DIR, exist_ok=True) 

app = FastAPI()
app.mount("/upload_img", StaticFiles(directory=IMG_DIR), name="upload_img")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

def generate_id():
    timestamp = int(time.time() * 1000)
    return str(timestamp)


class uploadResponse(BaseModel):
    similarity: float
    img_id: int

@app.get("/")
def root():
    return {"message": "Clip-imgServer"}


@app.post("/upload", response_model=uploadResponse)
def response_similarity(file: UploadFile = File(...), theme: str = Form(...)):
    try:
        print(f"file:{file.filename}, theme: {theme}")  
        img_id = generate_id()
        file_name = img_id + ".jpeg"
        file_path = os.path.join(IMG_DIR, file_name)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            image = Image.open(file_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file")



        inputs = PROCESSOR(text=[theme], images=image, return_tensors="pt", padding=True)

        outputs = MODEL(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        cosine_similarity = F.cosine_similarity(image_embeds, text_embeds)
        similarity_percentage = cosine_similarity.item() * 100
        print(f"cosine_similarity:{similarity_percentage}")  
        return uploadResponse(similarity=similarity_percentage,img_id = img_id)
    except Exception as e:
        raise HTTPException("error " + e)