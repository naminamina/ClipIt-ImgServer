from fastapi import FastAPI, File, UploadFile
import time
import os
import shutil 
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


IMG_DIR = "./upload_img"
os.makedirs(IMG_DIR, exist_ok=True) 

def generate_id():
    timestamp = int(time.time() * 1000)
    return str(timestamp)

class uploadResponse(BaseModel):
    similarity: float
    rank: int


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/upload", response_model=uploadResponse)
syance def response_similarity(file: UploadFile = File(...)):

    file_name = generate_id() + ".jpeg"
    file_path = os.path.join(IMG_DIR, file_name)
    print(file_name)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        image = Image.open(file_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")



    inputs = processor(text=["a photo of a cat"], images=image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    cosine_similarity = F.cosine_similarity(image_embeds, text_embeds)
    similarity_percentage = cosine_similarity.item() * 100

    rank = int(similarity_percentage // 10)
    return uploadResponse(similarity=similarity_percentage, rank=rank)





