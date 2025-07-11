import sys
import streamlit as st
from transformers import BertTokenizer
from tensorflow import keras
import numpy as np
from transformers import TFBertModel


print(sys.executable)


# モデルとトークナイザーの準備
model = keras.models.load_model(
    r"C:\Users\takus\testramen\ramen_retriever.h5",
    custom_objects={"TFBertModel": TFBertModel}
)
tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

# 推論のための関数
def tokenize_pair(q, a, max_len=64):
    inputs = tokenizer(q, a, padding='max_length', truncation=True, max_length=max_len, return_tensors="tf")
    return inputs['input_ids'][0], inputs['attention_mask'][0]

def get_best_answer(question, candidates):
    input_ids_list = []
    attention_masks_list = []
    for ans in candidates:
        ids, mask = tokenize_pair(question, ans)
        input_ids_list.append(ids)
        attention_masks_list.append(mask)
    input_ids = np.stack(input_ids_list)
    attention_masks = np.stack(attention_masks_list)
    preds = model.predict([input_ids, attention_masks])
    best_idx = np.argmax(preds)
    return candidates[best_idx], float(preds[best_idx][0])

# 候補文
candidates = [
    "スープは豚骨や鶏ガラを長時間煮込んで作ります。",
    "味噌ラーメンは北海道で生まれたラーメンです。",
    "チャーシューは豚バラ肉で作るトッピングです。",
    "ちぢれ麺はスープがよく絡む特徴があります。"
]

# Streamlit UI
st.title("🍜ラーメンプロフェッショナルAI")
st.markdown("### 気になるラーメンのこと、なんでも聞いてね♪")

user_question = st.text_input("🔍 質問を入力してね")

if user_question:
    answer, score = get_best_answer(user_question, candidates)
    st.markdown("### 💡 答え")
    st.success(f"{answer}")
    st.markdown(f"スコア：`{score:.4f}`")