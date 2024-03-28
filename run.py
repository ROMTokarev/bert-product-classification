import time
import torch
import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import AutoTokenizer

# config
config = AutoConfig.from_pretrained('model')
# tokenizer
tokenizer = AutoTokenizer.from_pretrained('model', pad_to_max_length=True)
# model
model = AutoModelForSequenceClassification.from_pretrained('model', config=config)

df = pd.read_csv('dataset.csv')
df.category_1.unique()
category_index = {i[1]:i[0] for i in enumerate(df.category_1.unique())}
category_index_reverce = {i[0]:i[1] for i in enumerate(df.category_1.unique())}

model.to('cpu')
model.eval()


while True:
    sku_title = input("Enter your value: ") 
    # print(val) 

    start_time = time.time()

    # sku_title = 'Эмаль для пола'
    tokens = tokenizer.encode(sku_title, add_special_tokens=True)
    tokens_tensor = torch.tensor([tokens])
    with torch.no_grad():
        logits = model(tokens_tensor)
    # Логиты по каждой категории
    logits = logits[0].detach().numpy()
    # Выбираем наиболее вероятную категорию товара
    predicted_class = np.argmax(logits, axis=1)
    print('\n')
    print(f'Предсказанная категория: {category_index_reverce[predicted_class[0]]}')
    print(f'Предсказанная категория: {predicted_class}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('\nЗатраченое время: ', elapsed_time)
    print('\n')