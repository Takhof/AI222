from tensorflow import keras
from transformers import BertTokenizer
import numpy as np
from transformers import TFBertModel


# モデル読み込み
model = keras.models.load_model(
    r"C:\Users\takus\testramen\ramen_retriever.h5",
    custom_objects={"TFBertModel": TFBertModel}
)


# トークナイザー準備
tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

# 質問
question = "ラーメンのスープってどう作るの？"

# 候補文たち
candidates = [
    "スープは豚骨や鶏ガラを長時間煮込んで作ります。",
    "味噌ラーメンは北海道で生まれたラーメンです。",
    "チャーシューは豚バラ肉で作るトッピングです。",
    "ちぢれ麺はスープがよく絡む特徴があります。"
]

# 推論準備
def tokenize_pair(q, a, max_len=64):
    inputs = tokenizer(q, a, padding='max_length', truncation=True, max_length=max_len, return_tensors="tf")
    return inputs['input_ids'][0], inputs['attention_mask'][0]

# 全候補をベクトル化
input_ids_list = []
attention_masks_list = []

for ans in candidates:
    ids, mask = tokenize_pair(question, ans)
    input_ids_list.append(ids)
    attention_masks_list.append(mask)

input_ids = np.stack(input_ids_list)
attention_masks = np.stack(attention_masks_list)

# モデルに食べさせる🍜
preds = model.predict([input_ids, attention_masks])

# 一番スコアが高い候補を選ぶ
best_idx = np.argmax(preds)
print("質問:", question)
print("候補の中でいちばん合ってる答えは…💡")
print("👉", candidates[best_idx])
print("スコア:", preds[best_idx][0])