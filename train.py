import os
import requests
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset
from typing import Dict, Optional

API_URL = "http://192.168.41.45:8081"
OLD_MODEL = "./codebert-base-mlm"
MODEL_DIR = "./checkpoint"
EVA_DATA_NUM = 3000
TRAIN_DATA_NUM = 12000
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "SmartBERT")


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
        res = session.get(url)
        res.raise_for_status()
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


def collect_data(start_id, data_num):
    count = 0
    data = []
    id = start_id
    while count < data_num:
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
    return data


def train():
    # Check if model directory exists and has checkpoints
    checkpoint = None
    if os.path.isdir(MODEL_DIR) and os.listdir(MODEL_DIR):
        checkpoints = [os.path.join(MODEL_DIR, d) for d in os.listdir(
            MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, d))]
        if checkpoints:
            # 获取最新的checkpoint
            checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"Resuming from checkpoint {checkpoint}")

    # Load model and tokenizer
    if checkpoint:
        model = RobertaForMaskedLM.from_pretrained(checkpoint)
    else:
        model = RobertaForMaskedLM.from_pretrained(OLD_MODEL)

    tokenizer = RobertaTokenizer.from_pretrained(OLD_MODEL)
    # Train dataset
    print("Train: collecting data")
    train_data = collect_data(1, TRAIN_DATA_NUM)
    print(f"Train data count: {len(train_data)}")
    train_inputs = tokenizer(train_data, return_tensors="pt",
                             padding="max_length", truncation=True, max_length=512)
    train_dataset = TrainDataset(train_inputs)

    # Evaluation dataset
    print("Evaluate: collecting data")
    eval_data = collect_data(20001, EVA_DATA_NUM)
    print(f"Eval data count: {len(eval_data)}")
    eval_inputs = tokenizer(eval_data, return_tensors="pt",
                            padding="max_length", truncation=True, max_length=512)
    eval_dataset = TrainDataset(eval_inputs)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # Define training args
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=32,
        save_steps=10000,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=10000,
        resume_from_checkpoint=checkpoint
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train(resume_from_checkpoint=checkpoint)

    # Save final model
    print(f"Saving final model to {FINAL_MODEL_PATH}")
    tokenizer.save_pretrained(FINAL_MODEL_PATH)
    trainer.save_model(FINAL_MODEL_PATH)


if __name__ == "__main__":
    train()
