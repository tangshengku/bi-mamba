#!/bin/bash

base_url="https://huggingface.co/datasets/cerebras/SlimPajama-627B/resolve/main/train/chunk1/"

for i in {3001..6000}
do
    url="${base_url}example_train_${i}.jsonl.zst"
    wget -nv "$url"
done