#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充 ./model/SmartBERT-v2 目录下缺失的 tokenizer.json 文件。
兼容 Roberta/CodeBERT/BERT 等模型结构。
"""

import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast

MODEL_DIR = "./model/SmartBERT-v2"
TOKENIZER_JSON_PATH = os.path.join(MODEL_DIR, "tokenizer.json")

def main():
    if os.path.exists(TOKENIZER_JSON_PATH):
        print(f"✅ 已存在: {TOKENIZER_JSON_PATH}")
        return

    print(f"⚙️ 未检测到 tokenizer.json，正在生成...")

    # 尝试加载 fast tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise RuntimeError("❌ 当前 tokenizer 不是 Fast 版本，无法生成 tokenizer.json。\n"
                           "请检查模型是否基于 BERT/Roberta 架构。")

    # 重新保存，会自动生成 tokenizer.json
    tokenizer.save_pretrained(MODEL_DIR)

    if os.path.exists(TOKENIZER_JSON_PATH):
        print(f"✅ 已生成: {TOKENIZER_JSON_PATH}")
    else:
        print("⚠️ 生成失败，请检查模型目录或尝试使用 RobertaTokenizerFast 重新加载。")

if __name__ == "__main__":
    main()
