# SmartBERT

🧐 Learning representations from **smart contracts**!

## Introduction

**SmartBERT** is a pre-trained programming language model based on [microsoft/codebert-base-mlm](https://huggingface.co/microsoft/codebert-base-mlm), which itself is built on the [RoBERTa](https://huggingface.co/facebook/roberta-base) architecture using a simple **Masked Language Model (MLM)** objective. **SmartBERT** is specifically fine-tuned for **smart contracts**, converting contract code into embeddings suitable for various downstream tasks in smart contract analysis.

## Installation

We recommend creating a virtual environment before installing the pip packages, such as with [Anaconda](https://www.anaconda.com/).

Then, install the requirements:

```bash
pip install -r requirements.txt
```

Download **SmartBERT** model from <https://github.com/web3se-lab/SmartBERT/releases> and unzip all the files to `/model`

## Usage

### Running the API

Start the API server with the following command:

```bash
./api.sh
```

### Running with Docker

Use the provided `docker-compose.yml` file to run SmartBERT with Docker.

```yml
version: "3"
services:
  smartbert:
    image: devilyouwei/smartbert:latest
    container_name: smartbert
    ports:
      - 8100:8100
```

## API

The following APIs are available.
Please use the POST method with JSON to request APIs.

### Tokenize

**Endpoint:** `http://localhost:8100/tokenize`

**Request:**

```json
{
  "text": "Smart Contract function-level code here..."
}
```

**Response:**

```json
[
  {
    "token": ["<s>", "//", "SP", "DX", "-", "License"], //...
    "ids": [0, 42326, 4186, 40190, 12] // ...
  }
  //...
]
```

### Embedding

Available poolings: average pooling, max pooling, CLS token pooling, and `pooler_output`

**Endpoint:** `http://localhost:8100/embedding`

**Request:**

```json
{
  "text": "Smart Contract function-level code here...",
  "pool": "avg"
}
```

- Param `text` can be string or string array of smart contract function code snippets
- Param `pool` can be `avg`, `max`, `cls`, `out`

**Response:**

```json
{
  "embedding": [
    [-0.006051725707948208, 0.10594873130321503, 0.07721099257469177] // ... dimension is 768
  ],
  "object": "embedding.avg"
}
```

## Retraining

To retrain SmartBERT, first download an original RoBERTa-based model:

```bash
cd base-model
git lfs install
git clone https://huggingface.co/microsoft/codebert-base-mlm
```

Modify the `OLD_MODEL="./base-model/{your initial model}"` variable in `train.py` to the path of the original model.

Then, run the training script:

```bash
./train.sh
```

## Setup

**SmartBERT V1** is trained on a dataset of more than **40,000** smart contracts by [Sen Fang](https://github.com/TomasAndersonFang), based on [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base)

From **V2**, we change the initial model to [CodeBERT-base-MLM](https://huggingface.co/microsoft/codebert-base-mlm).

**SmartBERT V2** is trained on a dataset of **12,000** smart contracts with 1,109,531 functions and evaluated on a distinct **3,000** contracts, maintaining a training-to-evaluation ratio of 4:1.
The total training step is **173370**.

This dataset for **V2** is specifically utilized in the **SmartIntentNN** project, which can be found at <https://github.com/web3se-lab/web3-sekit>.

**V3** will be trained on our full dataset of over **45,000** smart contracts.

## References

- [RoBERTa](https://huggingface.co/facebook/roberta-base)
- [CodeBERT-base-mlm](https://huggingface.co/microsoft/codebert-base-mlm)

## Contributors

Developed by **[Youwei Huang](https://www.devil.ren)**

Training by **[Sen Fang](https://github.com/TomasAndersonFang)** and **[Youwei Huang](https://www.devil.ren)**
