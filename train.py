import os
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset
import json

OLD_MODEL = "./base-model/roberta-base"
OUTPUT_DIR = "./checkpoint"
TRAIN_DATA = "./data/train.jsonl"  # 1, 16000 smart contracts
EVAL_DATA = "./data/eval.jsonl"  # 30000, 4000 smart contracts

EVA_DATA_NUM = 3000
TRAIN_DATA_NUM = 12000

FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "SmartBERT-roberta-16000")


class TrainDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {"input_ids": self.inputs["input_ids"][idx], "attention_mask": self.inputs["attention_mask"][idx]}


# Load dataset from json file
def dataset():
    try:
        with open(TRAIN_DATA, 'r', encoding='utf-8') as file:
            train = json.load(file)
        with open(EVAL_DATA, 'r', encoding='utf-8') as file:
            eval = json.load(file)
            return train, eval
    except FileNotFoundError:
        print("File not found.")
        return []
    except json.JSONDecodeError:
        print("Error decoding JSON.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def train():
    # Check if model directory exists and has checkpoints
    checkpoint = None
    if os.path.isdir(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
        checkpoints = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(
            OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
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

    train, eval = dataset()  # get dataset, train and eval
    # Train dataset
    print(f"Train data count: {len(train)}")
    train_inputs = tokenizer(train, return_tensors="pt",
                             padding="max_length", truncation=True, max_length=512)
    train_dataset = TrainDataset(train_inputs)

    # Evaluation dataset
    print(f"Eval data count: {len(eval)}")
    eval_inputs = tokenizer(eval, return_tensors="pt",
                            padding="max_length", truncation=True, max_length=512)
    eval_dataset = TrainDataset(eval_inputs)

    # MLM collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # Define training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=20,
        per_device_train_batch_size=64,
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
