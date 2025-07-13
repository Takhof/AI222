import tensorflow as tf
import json
from transformers import BertTokenizer, TFBertModel
import numpy as np



with open("retriever_qa.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

train_pairs = []

for item in qa_data:
    q = item["question"]
    train_pairs.append((q, item["positive"], 1))  # 正解ペア
    for neg in item["negatives"]:
        train_pairs.append((q, neg, 0))  # 間違いペア


#⃣Bert用にトークン化
tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

def tokenize_pair(q, a, max_len=64):
    inputs = tokenizer(q, a, padding='max_length', truncation=True, max_length=max_len, return_tensors="tf")
    return inputs['input_ids'][0], inputs['attention_mask'][0]

input_ids = []
attention_masks = []
labels = []

for q, a, label in train_pairs:
    ids, mask = tokenize_pair(q, a)
    input_ids.append(ids)
    attention_masks.append(mask)
    labels.append(label)

input_ids = tf.stack(input_ids)
attention_masks = tf.stack(attention_masks)
labels = tf.convert_to_tensor(labels)



#⃣モデル構築
bert_model = TFBertModel.from_pretrained("cl-tohoku/bert-base-japanese")

input_ids_in = tf.keras.Input(shape=(64,), dtype=tf.int32)
attention_mask_in = tf.keras.Input(shape=(64,), dtype=tf.int32)

bert_output = bert_model(input_ids=input_ids_in, attention_mask=attention_mask_in)[1]  # [CLS]出力
x = tf.keras.layers.Dense(256, activation='relu')(bert_output)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=[input_ids_in, attention_mask_in], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(
    [input_ids, attention_masks],
    labels,
    batch_size=8,
    epochs=3
)

model.save("ramen_retriever_light.h5", include_optimizer=False)
