import torch
import uvicorn
import os
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Union, List, Optional

API_PORT = int(os.environ.get("API_PORT", 9100))
API_MODEL = "./model/" + os.environ.get("API_MODEL", 'SmartBERT-V2-codebert')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: Union[List[str], str] = Field(...,
                                        description="Text or list of texts to be embedded")
    pool: Optional[str] = Field('avg', regex="^(cls|max|avg|out)$",
                                description="Optional pooling method: 'cls', 'max', 'avg', 'out'")


def convert_text_to_list(text: Union[List[str], str]) -> List[str]:
    if isinstance(text, str):
        return [text]
    return text


@app.post("/embedding")
async def embedding(request: TextRequest):
    global model, tokenizer
    texts = convert_text_to_list(request.text)
    pool_method = request.pool

    with torch.no_grad():
        inputs = tokenizer(texts, return_tensors="pt",
                           padding="max_length", truncation=True, max_length=512)

        inputs = {key: value.to(device) for key, value in inputs.items()}

        outputs = model(**inputs)

        if pool_method == 'cls':
            # CLS Token Pooling
            embedding_result = outputs.last_hidden_state[:, 0, :]
        elif pool_method == 'max':
            # Max Pooling
            embedding_result = outputs.last_hidden_state.max(1)[0]
        elif pool_method == 'avg':
            # mean pooling
            last_hidden_states = outputs.last_hidden_state
            attention_masks = inputs['attention_mask']
            attention_masks_expanded = attention_masks.unsqueeze(
                -1).expand(last_hidden_states.size()).float()
            sum_pooled = torch.sum(
                last_hidden_states * attention_masks_expanded, 1)
            embedding_result = sum_pooled / \
                torch.clamp(attention_masks_expanded.sum(1), min=1e-9)  # 避免除以0
        else:
            # Use pooler_output if available
            if hasattr(outputs, 'pooler_output'):
                embedding_result = outputs.pooler_output
            else:
                return {"error": "Model does not have pooler_output. Use 'cls', 'max', or 'avg' instead."}

        output = embedding_result.cpu().detach().numpy().tolist()

    # 清理显存
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"embedding": output, "object": f"embedding.{pool_method}"}


@app.post("/tokenize")
async def tokenize(request: TextRequest):
    global tokenizer
    texts = convert_text_to_list(request.text)
    tokenized_data = []

    for text in texts:
        with torch.no_grad():
            inputs = tokenizer(text, truncation=True,
                               padding="max_length", max_length=512)
            tokens = tokenizer.tokenize(text)
            tokens = [tokenizer.cls_token] + tokens + [tokenizer.eos_token]
            input_ids = inputs['input_ids']
            attention_masks = inputs['attention_mask']
            tokenized_data.append(
                {"token": tokens, "ids": input_ids, "masks": attention_masks}
            )

    return tokenized_data

if __name__ == "__main__":
    # check GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(API_MODEL)
    model = AutoModel.from_pretrained(API_MODEL).to(device)
    print("model:", API_MODEL)

    print("port:", API_PORT)
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
