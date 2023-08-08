import sys
import torch
import uvicorn
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware

device = f"cuda:{sys.argv[1]}" if torch.cuda.is_available() and len(
    sys.argv) >= 2 else "cpu"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/embedding")
async def embedding(text: str = Form(...)):
    global model, tokenizer, device
    print(text)
    input = tokenizer(text, truncation=True,
                      padding="max_length", max_length=512)['input_ids']
    print(input)
    embeddings = model(torch.tensor(input).to(device)[None, :])[0]
    output = embeddings.cpu().detach().numpy().tolist()
    return {"embedding": output[0]}


@app.post("/embeddingMax")
async def embeddingMax(text: str = Form(...)):
    global model, tokenizer, device
    print(text)
    input = tokenizer(text, truncation=True, max_length=512)['input_ids']
    print(input)
    embeddings = model(torch.tensor(input).to(device)[None, :])[0]
    output = embeddings.max(1)[0].cpu().detach().numpy().tolist()
    return {"embedding": output[0]}


@app.post("/embeddingAvg")
async def embeddingAvg(text: str = Form(...)):
    global model, tokenizer, device
    print(text)
    input = tokenizer(text, truncation=True, max_length=512)['input_ids']
    print(input)
    embeddings = model(torch.tensor(input).to(device)[None, :])[0]
    output = embeddings.mean(1).cpu().detach().numpy().tolist()
    return {"embedding": output[0]}


@app.post("/tokenize")
async def tokenize(text: str = Form(...)):
    global tokenizer
    print(text)
    input = tokenizer(text, truncation=True, padding="max_length",
                      max_length=512)['input_ids']
    print(input)
    token = tokenizer.tokenize(text)
    token = [tokenizer.cls_token] + token + [tokenizer.eos_token]
    print(token)
    return {"token": token, 'ids': input}


if __name__ == "__main__":

    # import bert model
    tokenizer = AutoTokenizer.from_pretrained("./model")
    model = AutoModel.from_pretrained("./model")

    print('device', device)
    model = model.to(device)

    uvicorn.run(app, host="0.0.0.0", port=8000)
