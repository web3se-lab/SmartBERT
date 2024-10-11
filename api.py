import torch
import uvicorn
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Union, List, Optional

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求体的数据模型
class TextRequest(BaseModel):
    text: Union[List[str], str] = Field(..., description="Text or list of texts to be embedded")
    pool: Optional[str] = Field('avg', regex="^(cls|max|avg|out)$", description="Optional pooling method: 'cls', 'max', 'avg', 'out'")

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
        inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        input_ids, attention_mask = inputs['input_ids'].to("cuda"), inputs['attention_mask'].to("cuda")

        outputs = model(input_ids, attention_mask=attention_mask)

        if pool_method == 'cls':
            # CLS Token Pooling
            embedding_result = outputs.last_hidden_state[:, 0, :]
        elif pool_method == 'max':
            # Max Pooling
            embedding_result = outputs.last_hidden_state.max(1)[0]
        elif pool_method == 'avg':
            # Mean Pooling
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size())
            sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask_expanded, 1)
            sum_mask = attention_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)  # 避免除0错误
            embedding_result = sum_embeddings / sum_mask
        else:
            # Use pooler_output if available
            if hasattr(outputs, 'pooler_output'):
                embedding_result = outputs.pooler_output
            else:
                return {"error": "Model does not have pooler_output. Use 'cls', 'max', or 'avg' instead."}

        output = embedding_result.cpu().detach().numpy().tolist()

    # 清理显存
    torch.cuda.empty_cache()

    return {"embedding": output, "object": f"embedding.{pool_method}"}

@app.post("/tokenize")
async def tokenize(request: TextRequest):
    global tokenizer
    texts = convert_text_to_list(request.text)
    tokenized_data = []

    for text in texts:
        with torch.no_grad():
            inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512)
            tokens = tokenizer.tokenize(text)
            tokens = [tokenizer.cls_token] + tokens + [tokenizer.eos_token]
            input_ids = inputs['input_ids']
            tokenized_data.append({"token": tokens, "ids": input_ids})

    return tokenized_data

if __name__ == "__main__":
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./model")
    model = AutoModel.from_pretrained("./model").to("cuda")  # 加载模型并移动到GPU

    uvicorn.run(app, host="0.0.0.0", port=9100)