import os
import time
import shutil 
import logging  
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO

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

@app.get("/")
def root():
    return {"message": "Clip-imgServer"}


@app.post("/upload", response_model=uploadResponse)

def response_similarity(theme: str = Form(...), img_url: str = Form(...)):

    logging.info(f"images:{img_url}, theme: {theme}")  
    img_data = requests.get(img_url)
    img_data = Image.open(BytesIO(img_data.content))
    inputs = PROCESSOR(text=[theme], images=[img_data], return_tensors="pt", padding=True)

    outputs = MODEL(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    cosine_similarity = F.cosine_similarity(image_embeds, text_embeds)
    similarity_percentage = cosine_similarity.item() * 100
    logging.info(f"cosine_similarity:{similarity_percentage}")  
    return uploadResponse(similarity=similarity_percentage)


