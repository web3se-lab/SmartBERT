import torch
import numpy
import uvicorn
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# import bert model
tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModel.from_pretrained("./model")


@app.post("/embedding")
async def embedding(text: str = Form(...)):
    print(text)
    input = tokenizer(text, truncation=True,
                      padding="max_length", max_length=512)['input_ids']
    print(input)
    embeddings = model(torch.tensor(input)[None, :])[0]
    output = embeddings.detach().numpy().tolist()
    return {"embedding": output[0]}


@app.post("/embeddingMax")
async def embeddingMax(text: str = Form(...)):
    print(text)
    input = tokenizer(text, truncation=True, max_length=512)['input_ids']
    print(input)
    embeddings = model(torch.tensor(input)[None, :])[0]
    output = embeddings.max(1)[0].detach().numpy().tolist()
    return {"embedding": output[0]}


@app.post("/embeddingAvg")
async def embeddingAvg(text: str = Form(...)):
    print(text)
    input = tokenizer(text, truncation=True, max_length=512)['input_ids']
    print(input)
    embeddings = model(torch.tensor(input)[None, :])[0]
    output = embeddings.mean(1).detach().numpy().tolist()
    return {"embedding": output[0]}


@app.post("/tokenize")
async def tokenize(text: str = Form(...)):
    print(text)
    input = tokenizer(text, truncation=True, padding="max_length",
                      max_length=512)['input_ids']
    print(input)
    token = tokenizer.tokenize(text)
    token = [tokenizer.cls_token] + token + [tokenizer.eos_token]
    print(token)
    return {"token": token, 'ids': input}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                log_level="info", reload=True)
