from factory import Factory
from model.model import Bert_Model, SBert, TBert, LayerNorm, Output_Layer

f = Factory()
fit = {}
score = {}
confident = 0.6

#===========================================API========================================================================

import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from pydantic import BaseModel
from typing import Optional

app = FastAPI()
class Question(BaseModel):
    question: str

class FilePath(BaseModel):
    path: str

@app.post("/duplicate/find-duplicate")
async def create_item(request: Request):
    dataRequest = await request.json()
    fit,response = f.find_duplicate([dataRequest['question']],confident, subject=dataRequest['subject'])
    error = ""
    if response == None:
        return {
            error: "error...!"
        }
    return response

@app.post("/duplicate/train-data")
async def train_data(file_path: FilePath):
    print(file_path.path)
    f.save_encoded(file_path.path)
    return "training successful...!"


@app.post("/duplicate/train-all-data")
async def train_data_all(request: Request):
    dataRequest = await request.json()
    f.save_encoded_all(dataRequest)
    return "training successful...!"

@app.post("/duplicate/train-data-subject")
async def train_data_subject(request: Request):
    dataRequest = await request.json()
    f.save_encoded_with_subject(dataRequest)
    return "training successful...!"

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)