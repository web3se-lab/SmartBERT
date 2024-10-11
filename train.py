import requests
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset
from typing import Dict, Optional


API_URL = "http://192.168.41.45:8081"
OLD_MODEL = "./codebert-base-mlm"
NEW_MODEL = "./new_model"
TRAIN_DATA_NUM = 12000
EVA_DATA_NUM = 3000


class TrainDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {"input_ids": self.inputs["input_ids"][idx], "attention_mask": self.inputs["attention_mask"][idx]}


# Get data for training
session = requests.Session()


def get_train_data(id: int) -> Optional[Dict[str, list]]:
    try:
        url = f"{API_URL}/data/get?key={id}"
        print(f"Request: {url}")

        # 从 API 获取数据集
        res = session.get(url)
        res.raise_for_status()  # 检查 HTTP 请求是否成功

        data = res.json()
        return data.get('CodeTree', None) if data else None

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except ValueError as e:
        print(f"JSON decode error: {e}")
    except KeyError as e:
        print(f"Key error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    return None


def train():
    # define model and tokenizer
    model = RobertaForMaskedLM.from_pretrained(OLD_MODEL)
    tokenizer = RobertaTokenizer.from_pretrained(OLD_MODEL)

    # train dataset: 1, 12000
    count = 0
    data = []
    id = 1
    print("Train: collecting data")
    while count < TRAIN_DATA_NUM:
        try:
            code = get_train_data(id)
            if code is None:
                print(f"ID: {id} - No data found")
                id += 1
                continue

            for key in code:
                data.extend(code[key].values())

            count += 1
            print(f"ID: {id} - Data collected: {len(data)} - Count: {count}")

        except Exception as e:
            print(f"ID: {id} - Error occurred: {e}")

        id += 1

    print(f"Train data count: {len(data)}")
    inputs = tokenizer(data, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=512)
    train_dataset = TrainDataset(inputs)

    # Evaluation dataset: 20001, 3000
    data = []
    count = 0
    id = 20001
    print("Evaluate: collecting data")
    while count < EVA_DATA_NUM:
        try:
            code = get_train_data(id)
            if code is None:
                print(f"ID: {id} - No data found")
                id += 1
                continue

            for key in code:
                data.extend(code[key].values())

            count += 1

            print(f"ID: {id} - Data collected: {len(data)} - Count: {count}")

        except Exception as e:
            print(f"ID: {id} - Error occurred: {e}")

        id += 1

    print(f"Eval data count: {len(data)}")
    inputs = tokenizer(data, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=512)
    eval_dataset = TrainDataset(inputs)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # define training args
    training_args = TrainingArguments(
        output_dir=NEW_MODEL,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        save_steps=10000,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=10000,
    )

    # 定义 Trainer 对象
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()


train()
