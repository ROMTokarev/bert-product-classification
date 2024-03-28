import time
import numpy as np
import pandas as pd
from random import randint
import torch
from torch.utils.data import (TensorDataset,
                              DataLoader,
                              RandomSampler)

from keras.preprocessing.sequence import pad_sequences

from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SequentialSampler

import warnings
warnings.filterwarnings("ignore")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

df = pd.read_csv('dataset.csv')
# print(df.shape)
# print(df.sample(5))

df.category_1.unique()

category_index = {i[1]:i[0] for i in enumerate(df.category_1.unique())}
category_index_reverce = {i[0]:i[1] for i in enumerate(df.category_1.unique())}

# Переведём все метки датасета в числа
sentences = df.name.values
labels = [category_index[i] for i in df.category_1.values]

assert len(sentences) == len(labels) == df.shape[0]

sentences = ['[CLS] ' + sentence + ' [SEP]' for sentence in sentences]

train_sentences, test_sentences, train_category, test_category = train_test_split(sentences, labels, test_size=0.005)

# print(len(train_sentences), len(test_sentences))

tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'DeepPavlov/rubert-base-cased', trust_repo=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in train_sentences]
# print(tokenized_texts[42])

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# Соберём все размеры последовательностей
lenths = [len(sent) for sent in tokenized_texts]
# plt.hist(lenths)
# time.sleep(999)

input_ids = pad_sequences(
    input_ids,
    # максимальная длина предложения
    maxlen=24,
    dtype='long',
    truncating='post',
    padding='post'
)

# print(input_ids[42])

attention_masks = [[float(i>0) for i in seq] for seq in input_ids]
# print(attention_masks[42])

# assert len(input_ids[42]) == len(attention_masks[42])

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(
    input_ids, train_category,
    random_state=42,
    test_size=0.1
)

train_masks, validation_masks, _, _ = train_test_split(
    attention_masks,
    input_ids,
    random_state=42,
    test_size=0.1
)

assert len(train_inputs) == len(train_labels) == len(train_masks)
assert len(validation_inputs) == len(validation_labels) == len(validation_masks)

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)

validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(
    train_data,
    # Данные по батчам разбиваем произвольно с помощью RandomSampler
    sampler=RandomSampler(train_data),
    batch_size=64
)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_dataloader = DataLoader(
    validation_data,
    sampler=SequentialSampler(validation_data),
    batch_size=64
)

config = AutoConfig.from_pretrained('DeepPavlov/rubert-base-cased',
                                    num_labels=len(category_index),
                                    id2label=category_index_reverce,
                                    label2id=category_index)

model = AutoModelForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased', config=config)

model.to(device)

param_optimizer = list(model.named_parameters())

# Можно посмотреть или изменить. Но нам этого не нужно, инициализируем лишь функцию
# оптимизации. В качестве оптимизатора будем использовать оптимизированный
# Adam (adaptive moment estimation)
# for name, _ in param_optimizer:
#     print(name)

no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

train_loss_set = []
train_loss = 0

start_time = time.time()
# Переводим модель в training mode
model.train()

for step, batch in enumerate(train_dataloader):
    # Переводим данные на видеокарту
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    # Обнуляем градиенты
    optimizer.zero_grad()

    # Прогоняем данные по слоям нейросети
    loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

    train_loss_set.append(loss[0].item())

    # Обратный прогон
    loss[0].backward()

    # Шаг
    optimizer.step()

    # Обновляем loss
    train_loss += loss[0].item()
    #print(f'Loss: {loss[0].item()}')
end_time = time.time()
 
# разница между конечным и начальным временем
elapsed_time = end_time - start_time
print('*'*20)
print('\n\nВремя обучения: ', elapsed_time)
print(f'Лосс на обучении: {train_loss / len(train_dataloader)}')
print('*'*20)


# plt.plot(train_loss_set)
# plt.title("Loss на обучении")
# plt.xlabel("Батчи")
# plt.ylabel("Потери")
# plt.show()


start_time = time.time()
# Переводим модель в evaluation mode
model.eval()

valid_preds, valid_labels = [], []

for batch in validation_dataloader:
    # добавляем батч для вычисления на GPU
    batch = tuple(t.to(device) for t in batch)

    b_input_ids, b_input_mask, b_labels = batch



    # Вычислять градиенты не нужно
    with torch.no_grad():
        logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    # Перемещаем логиты и метки на CPU
    logits = logits[0].detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    batch_preds = np.argmax(logits, axis=1)
    batch_labels = label_ids #np.concatenate(label_ids)
    valid_preds.extend(batch_preds)
    valid_labels.extend(batch_labels)

# print(classification_report(valid_labels, valid_preds, target_names=category_index_reverce.values()))

model.save_pretrained('model')
tokenizer.save_pretrained('model')

end_time = time.time()
elapsed_time = end_time - start_time
print('Время на мастеринг: ', elapsed_time)
print('\n')

# print(category_index)
