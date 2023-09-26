import requests
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset


API_URL = "http://192.168.41.46:8081"
OLD_MODEL = "./model"
NEW_MODEL = "./new_model"


class TrainDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {"input_ids": self.inputs["input_ids"][idx], "attention_mask": self.inputs["attention_mask"][idx]}


def get_train_data(id: int) -> str:
    try:
        url = f"{API_URL}/data/vulnerability?key={id}"
        print(f"Request: {url}")

        # get dataset from API
        res = requests.get(url)

        if res.status_code == 200:
            data = res.json()
            if data:
                return data['tree']
            else:
                return None
        else:
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def train():
    # define model and tokenizer
    model = RobertaForMaskedLM.from_pretrained(OLD_MODEL)
    tokenizer = RobertaTokenizer.from_pretrained(OLD_MODEL)

    data = []
    # iterate dataset
    print(f"Train: 1-2000")
    for id in range(1, 2001):
        code = get_train_data(id)
        print(f"ID: {id}")
        if code is None:
            continue

        for i in code:
            for j in code[i]:
                data.append(code[i][j])

    print(f"Train data count: {len(data)}")
    inputs = tokenizer(data, return_tensors="pt",
                       padding="max_length", truncation=True, max_length=512)
    train_dataset = TrainDataset(inputs)

    # # 创建数据加载器
    # batch_size = 1  # 指定批处理大小
    # train_loader = DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True)

    # # 遍历和打印每一行数据集
    # for batch in train_loader:
    #     # batch是一个包含多个样本的字典，包含了"input_ids"和"attention_mask"字段
    #     input_ids = batch["input_ids"]
    #     attention_mask = batch["attention_mask"]
    #     # 打印当前批次的数据
    #     for i in range(len(input_ids)):
    #         print(f"Sample {i+1}:")
    #         print("input_ids:", input_ids[i])
    #         print("attention_mask:", attention_mask[i])

    data = []
    # iterate dataset
    print(f"Evaluate: 2001-2503")
    for id in range(2001, 2504):
        code = get_train_data(id)
        print(f"ID: {id}")
        if code is None:
            continue

        for i in code:
            for j in code[i]:
                data.append(code[i][j])

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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()


train()
