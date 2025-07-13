import sys
import streamlit as st
from transformers import BertTokenizer
from tensorflow import keras
import numpy as np
from transformers import TFBertModel
import os
import pandas as pd
import json
from pathlib import Path
import gdown

print(sys.executable)


# q/a dataのCSV
data_path = Path("qa_data.csv")
if not data_path.exists():
    df = pd.DataFrame(columns=["question", "answer"])
    df.to_csv(data_path, index=False, encoding="utf-8-sig")


# モデルとトークナイザーの準備
model_path = "ramen_retriever.h5"
if not os.path.exists(model_path):
    file_id = "1CmJuR_H2eFBaGY88XdPdrLcoYcvqwt7T"  # 自分のファイルIDにしてね
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
model = keras.models.load_model(model_path, custom_objects={"TFBertModel": TFBertModel})
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



st.markdown("### ✍️ 回答データの追加")

new_q = st.text_input("質問を入力してね（追加用）", key="add_q")
new_a = st.text_input("その答えを入力してね", key="add_a")

if st.button("データを追加する"):
    if new_q and new_a:
        df = pd.read_csv(data_path)
        df = pd.concat([df, pd.DataFrame([{"question": new_q, "answer": new_a}])], ignore_index=True)
        df.to_csv(data_path, index=False, encoding="utf-8-sig")
        st.success("✅ データを追加したよ！ありがとう")
    else:
        st.warning("⚠️ 質問と答え、どっちもいれてね〜！")


if st.button("今までの追加データを見る"):
    df = pd.read_csv(data_path)
    if df.empty:
        st.info("まだ追加されたデータはないみたい〜💤")
    else:
        for i, row in df.iterrows():
            st.markdown(f"**{i+1}. Q:** {row['question']}")
            st.markdown(f"A: {row['answer']}")

