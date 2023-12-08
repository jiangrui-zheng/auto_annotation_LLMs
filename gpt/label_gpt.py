import pandas as pd
import openai
from tqdm import tqdm
import re
import torch
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def get_gpt_label(content, sentence):
    if pd.isnull(sentence):
        return None
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You're a nlp expert helping me categorize hate speech sentences"},
            # {"role": "user", "content": content + "\n\n- text: {}\n- label:".format(sentence)},
            {"role": "user", "content": content + "\n\nQuestion:["+sentence + "]"},
        ],
        max_tokens = 50
        )
    result = response.choices[0].message.content
    return result


with open('/data/jzheng36/auto_annotation/prompt.txt', 'r') as file:
    content = file.read()

test_cases = pd.read_csv("/data/jzheng36/auto_annotation/test_dataset.csv", sep ="\t")

labels = []
for index, row in tqdm(test_cases.iterrows(), total=test_cases.shape[0]):
    label = None
    attempts = 0
    while not label and attempts < 10:
        label = get_gpt_label(content, row['text'])
        match = re.search(r'\d+', label)
        label = match.group(0) if match else None
        if not label:
            attempts += 1
            time.sleep(5)
    labels.append(label)

test_cases['prediction'] = labels

test_cases.to_csv("/data/jzheng36/auto_annotation/test_with_gpt.csv", sep="\t", index=False)