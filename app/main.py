import os
import logging  
import requests
from io import BytesIO
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from PIL import Image

os.environ["TF_CACHE"] = "/tmp"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
HF_MODEL_PATH = 'line-corporation/clip-japanese-base'


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# def clip_analysis(theme, img_url):
#     try:
#         img_request_data = requests.get(img_url)
#         img_open_data = Image.open(BytesIO(img_request_data.content))
#         inputs = PROCESSOR(text=[theme], images=[img_open_data], return_tensors="pt", padding=True)

#         outputs = MODEL(**inputs)
#         image_embeds = outputs.image_embeds
#         text_embeds = outputs.text_embeds

#         cosine_similarity = F.cosine_similarity(image_embeds, text_embeds)
#         percentage_similarity = round(cosine_similarity.item() * 100, 2)
#         return cosine_similarity
#     except Exception  as e:
#         logging.info(F"Clip error:{e}")


def clip_analysis(theme, img_url):
    try:
        img_data = requests.get(img_url)
        img_data = Image.open(BytesIO(img_data.content))
        # inputs = PROCESSOR(text=[theme], images=[img_data], return_tensors="pt", padding=True)

        device = "cpu"
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)
        processor = AutoImageProcessor.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(HF_MODEL_PATH, trust_remote_code=True).to(device)

        image = processor(img_data, return_tensors="pt").to(device)
        text = tokenizer([theme]).to(device)
        image_features = model.get_image_features(**image)
        text_features = model.get_text_features(**text)
        cosine_similarity = F.cosine_similarity(image_features, text_features)
        percentage_similarity = round(cosine_similarity.item() * 100, 2)

        return percentage_similarity
    except Exception  as e:
        logging.info(F"Clip error:{e}")
        
class uploadResponse(BaseModel):
    similarity: float

@app.get("/")
def root():
    return {"message": "Clip-imgServer"}


@app.post("/upload", response_model=uploadResponse)

def response_similarity(theme: str = Form(...), img_url: str = Form(...)):
    try:

        logging.info(f"theme: {theme}, images:{img_url}")  
        return_similarity = clip_analysis(theme, img_url)
        logging.info(f"cosine_similarity:{return_similarity}")  
        return uploadResponse(similarity=return_similarity)


    except Exception  as e:
        logging.info(F"Server error:{e}")
        raise e