# SmartBERT

Learning representations from **smart contracts**!

## Install

```bash
pip install -r requirements.txt
```

## Model

```bash
wget https://github.com/web3se-lab/SmartBERT/releases/download/v1.0.0/SmartBERT_model_v1.zip
```

Then extract all the files to the dir `model`

## Run API

```bash
python ./api.py
```

## For Docker

You can also start SmartBERT API with docker and docker-compose

**docker-compose.yml**

```yml
version: '3'
services:
    smartbert:
        image: devilyouwei/smartbert:latest
        container_name: smartbert
        ports:
            - 8100:8100
```

## API Documentation

### POST: [http://localhost:8100/tokenize](http://localhost:8100/tokenize)

**Request:**

```json
{
    "text": "//SPDX-License-Identifier: MIT ..."
}
```

**Response:**

```json
{
    "token": [ "<s>", "//", "SP", "DX", "-", "License", ...  ],
    "ids": [ 0, 42326, 4186, 40190, 12, ...  ]
}
```

### POST: [http://localhost:8100/embedding](http://localhost:8100/embedding)

**Request:**

```json
{
    "text": "//SPDX-License-Identifier: MIT ..."
}
```

**Response:**

```json
{
    "embedding": [
        [ -0.09258933365345001, ...],
        [ ..., 0.21117761731147766 ]
    ],
    "object": "embedding"
}
```

### POST: [http://localhost:8100/embedding-avg](http://localhost:8100/embedding-avg)

**Request:**

```json
{
    "text": "//SPDX-License-Identifier: MIT ..."
}
```

**Response:**

```json
{
    "embedding": [
        -0.006051725707948208,
        0.10594873130321503,
        ...
        0.07721099257469177
    ],
    "object": "embedding.avg"
}
```

### POST: [http://localhost:8100/embedding-max](http://localhost:8100/embedding-max)

**Request:**

```json
{
    "text": "//SPDX-License-Identifier: MIT ..."
}
```

**Response:**

```json
{
    "embedding": [
        0.7459375858306885,
        1.428691029548645,
        ...
    ],
    "object": "embedding.max"
}
```

## Thanks

Trained by [Sen Fang](https://github.com/TomasAndersonFang)

Developed by [Youwei Huang](https://github.com/devilyouwei)

Powered by RoBERTa
