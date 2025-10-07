import torch
from transformers import AutoTokenizer, AutoModel
import os

model_dir = "./model/SmartBERT-v2"
onnx_dir = os.path.join(model_dir, "onnx")
os.makedirs(onnx_dir, exist_ok=True)

print(f"Loading model from {model_dir} ...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir)
model.eval()

dummy_input = tokenizer("function add(uint a, uint b) { return a + b; }", return_tensors="pt")

onnx_path = os.path.join(onnx_dir, "model.onnx")
print(f"Exporting model to {onnx_path} ...")

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    opset_version=13,
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
    },
)

print("✅ Export successful!")
print(f"ONNX file saved at: {onnx_path}")
