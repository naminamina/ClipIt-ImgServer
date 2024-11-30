import os
import logging  
import requests
from io import BytesIO
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoProcessor
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL = AutoModel.from_pretrained("google/siglip-base-patch16-256-multilingual")
PROCESSOR = AutoProcessor.from_pretrained("google/siglip-base-patch16-256-multilingual")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clip_analysis(theme, img_url):
    try:
        img_request_data = requests.get(img_url)
        img_open_data = Image.open(BytesIO(img_request_data.content))

        texts = [theme]

        inputs = PROCESSOR(text=texts, images=img_open_data, padding="max_length", return_tensors="pt")
        outputs = MODEL(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = torch.sigmoid(logits_per_image) 
        percentage_similarity = round(probs.item() * 100, 2)

        return percentage_similarity
    except Exception  as e:
        logging.info(F"Clip error:{e}")
        raise HTTPException(status_code=500, detail="Clip error.")


class uploadResponse(BaseModel):
    similarity: float
class RequestModel(BaseModel):
    img_url: str
    theme: str
@app.get("/")
def root():
    return {"message": "Clip-imgServer"}

@app.post("/upload", response_model=uploadResponse)

def response_similarity(theme: str = Form(...), img_url: str = Form(...)):
    try:
        if not theme or not img_url:
            raise HTTPException(status_code=400, detail="request error. thmemeとimg_urlが必要です")
        logging.info(f"theme: {theme}, images:{img_url}")  
        return_similarity = clip_analysis(theme, img_url)
        logging.info(f"cosine_similarity:{return_similarity}")  
        return uploadResponse(similarity=return_similarity)

    except ValidationError as e:
        logging.info(F"Validation error:{e}")
        raise HTTPException(status_code=422, detail=f"Validation error: {str(e)}")

    except Exception  as e:
        logging.info(F"Server error:{e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")
