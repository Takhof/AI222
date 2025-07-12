import tensorflow as tf
import json
import numpy as np
from keras.utils import custom_object_scope
from keras.utils import get_custom_objects
from transformers import BertTokenizer
from transformers import TFBertModel
import pandas as pd

# ✅ モデルをロード（

model = tf.keras.models.load_model(
    "ramen_retriever.h5",
    custom_objects={"TFBertModel": TFBertModel}
)


# ✅ トークナイザ（BERT日本語）
tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

# ✅ Q&Aデータを読み込み（CSVから！）
df = pd.read_csv("qa_data.csv")
train_pairs = [(row["question"], row["answer"], 1) for _, row in df.iterrows()]

# ✅ トークン化関数
def tokenize_pair(q, a, max_len=64):
    inputs = tokenizer(q, a, padding='max_length', truncation=True, max_length=max_len, return_tensors="tf")
    return inputs['input_ids'][0], inputs['attention_mask'][0]

# ✅ トークナイズ処理
input_ids, attention_masks, labels = [], [], []
for q, a, label in train_pairs:
    ids, mask = tokenize_pair(q, a)
    input_ids.append(ids)
    attention_masks.append(mask)
    labels.append(label)

input_ids = tf.stack(input_ids)
attention_masks = tf.stack(attention_masks)
labels = tf.convert_to_tensor(labels)

# ✅ 追加学習スタートっ！
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(
    [input_ids, attention_masks],
    labels,
    batch_size=8,
    epochs=2
)

# ✅ 保存しなおし！
model.save("ramen_retriever_finetuned", save_format="tf")
print("🍜💕 モデルにラーメンQ&Aを追加学習したよ〜♡")