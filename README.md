# SmartBERT

🧐 Learning representations from **smart contracts**!

![SmartBERT Framework](./framework.png)

## Introduction

**SmartBERT** is a pre-trained programming language model specifically fine-tuned for analyzing **smart contracts**. Based on [microsoft/codebert-base-mlm](https://huggingface.co/microsoft/codebert-base-mlm), which itself follows the [RoBERTa](https://huggingface.co/facebook/roberta-base) architecture using a simple **Masked Language Model (MLM)** objective, SmartBERT converts contract code into embeddings suitable for various downstream tasks in smart contract analysis.

## Installation

To set up the environment for SmartBERT, it is recommended to first create a virtual environment using tools like [Anaconda](https://www.anaconda.com/).

Then, install the required packages with:

```bash
pip install -r requirements.txt
```

Download the **SmartBERT** models from <https://github.com/web3se-lab/SmartBERT/releases> and unzip all files into the `/model` directory, e.g., `/model/SmartBERT-codebert-16000`.

Or, you can download **SmartBERT** from 🤗 Hugging Face: <https://huggingface.co/web3se/SmartBERT-v2>.

## Usage

### Running the API

Start the API server using the command:

```bash
./api.sh
```

### Running with Docker

You can also run SmartBERT using Docker with the provided `docker-compose.yml` file:

```yml
version: "3"
services:
  smartbert:
    image: devilyouwei/smartbert:latest
    container_name: smartbert
    ports:
      - 8100:8100
```

## API Documentation

Please always input smart contract _function-level_ code into **SmartBERT**.

### Tokenize

**Endpoint:** `http://localhost:9100/tokenize`

Use the **POST** method with **JSON** to tokenize text.

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
    "token": ["<s>", "//", "SP", "DX", "-", "License"],
    "ids": [0, 42326, 4186, 40190, 12],
    "masks": [1, 1, 1, 1, 0]
  }
]
```

### Embedding

Available pooling methods: average pooling, max pooling, CLS token pooling, and `pooler_output`.

**Endpoint:** `http://localhost:9100/embedding`

**Request:**

```json
{
  "text": ["Smart Contract function-level code here..."],
  "pool": "avg"
}
```

- `text`: A string or an array of strings containing smart contract function code snippets.
- `pool`: Can be one of `avg`, `max`, `cls`, `out`.

**Response:**

```json
{
  "embedding": [
    [-0.006051725707948208, 0.10594873130321503, 0.07721099257469177]
  ],
  "object": "embedding.avg"
}
```

- `embedding`: Contains 768-dimensional vectors representing the input texts.

## Retraining

To retrain SmartBERT, first download the original RoBERTa-based model:

```bash
cd base-model
git lfs install
git clone https://huggingface.co/microsoft/codebert-base-mlm
```

Update the `OLD_MODEL` variable in `train.py` to point to your original model directory.

Place the dataset files `train.jsonl` (for training) and `eval.jsonl` (for evaluation) in the `/data` directory.

Then, initiate the training process by running:

```bash
./train.sh
```

Further customization of training settings can be done in `train.py`.

## Setup

### SmartBERT Versions

- **V1**: Trained on a dataset of over **40,000** smart contracts based on [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base).
- **V2**: The initial model is changed to [CodeBERT-base-MLM](https://huggingface.co/microsoft/codebert-base-mlm) and trained on **16,000** smart contracts. Refer to the package `README.md` for more details.
- **V3**: Upcoming version trained on a full dataset of over **40,000** smart contracts.

The **V2** dataset is also utilized in the **SmartIntentNN** project, which can be found at <https://github.com/web3se-lab/web3-sekit>.

## References

- [RoBERTa](https://huggingface.co/facebook/roberta-base)
- [CodeBERT-base-mlm](https://huggingface.co/microsoft/codebert-base-mlm)

## Contributors

Developed by **[Youwei Huang](https://www.devil.ren)**

Trained by **[Sen Fang](https://github.com/TomasAndersonFang)** and **[Youwei Huang](https://www.devil.ren)**

## Acknowledgments

- [Institute of Intelligent Computing Technology, Suzhou, CAS](http://iict.ac.cn/)
