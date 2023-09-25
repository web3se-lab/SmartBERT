import requests
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

URL = "http://192.168.41.46:8081"
OLD_MODEL = "./model"
NEW_MODEL = "./new_model"
GPU = "cuda:2"


class TrainDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {"input_ids": self.inputs["input_ids"][idx], "attention_mask": self.inputs["attention_mask"][idx]}


def get_train_data(id: int) -> str:
    try:
        # 构建请求URL，增加key参数
        url = f"{URL}/data/vulnerability?key={id}"
        print(f"Request: {url}")

        # 发起GET请求
        res = requests.get(url)

        # 检查响应状态码
        if res.status_code == 200:
            # 解析JSON响应并返回
            data = res.json()
            if data:
                return data['tree']
            else:
                return None
        else:
            return None

    except Exception as e:
        # 处理请求错误，例如连接错误等
        print(f"Error: {e}")
        return None


def train():
    # define model and tokenizer
    model = RobertaForMaskedLM.from_pretrained(OLD_MODEL)
    tokenizer = RobertaTokenizer.from_pretrained(OLD_MODEL)
    device = torch.device(GPU if torch.cuda.is_available() else "cpu")

    model.to(device)

    data = []
    # iterate dataset
    for id in range(1, 2):
        code = get_train_data(id)
        print(f"ID: {id}")
        if code is None:
            continue

        for i in code:
            for j in code[i]:
                data.append(code[i][j])

    inputs = tokenizer(data, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=512)
    dataset = TrainDataset(inputs)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # define training args
    training_args = TrainingArguments(
        output_dir=NEW_MODEL,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        save_steps=100,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=100,
    )

    # 定义 Trainer 对象
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    trainer.train()


train()
