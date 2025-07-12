# 🍜 Ramen Retriever AI

日本語BERTベースで構築した、ラーメン専門の質問応答モデル！  
質問に対して、もっとも適切なラーメン情報を高精度にマッチングするAIです✨  

---

## 🚀 機能概要

- 🤖 [`cl-tohoku/bert-base-japanese`](https://huggingface.co/cl-tohoku/bert-base-japanese) によるBERTベースモデルを使用  
- 💡 質問と回答のペアを正解（1）・不正解（0）として分類するバイナリ分類モデル  
- 🔀 追加のQ&Aで簡単に再学習・Fine-tuning可能  
- 🧠 StreamlitやAPIへの応用もカンタン！  

---

## 📜 ディレクトリ構成

```
ramen-retriever-ai/
├── model_creator.py         # 最初のモデル学習用スクリプト
├── model_finetune.py        # 追加学習用スクリプト
├── retriever_qa.json        # 質問・正解・誤答データセット
├── qa_data.json             # 追加学習用のQ&Aデータ（question/answer形式）
├── ramen_model.h5           # 学習済みKerasモデル
├── requirements.txt         # 依存パッケージリスト
└── README.md                # このファイル
```

---

## 🛠️ セットアップ方法

### 1. Python環境の準備

```bash
python -m venv venv
source venv/bin/activate  # Windowsの方は venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🔪 モデル学習（初回）

`retriever_qa.json` を以下の形式で用意してください：

```json
[
  {
    "question": "ラーメンのスープの種類は？",
    "positive": "醤油、味噌、塩、とんこつなどがあります。",
    "negatives": [
      "サッカーは11人でやります。",
      "ペンギンは飛べません。"
    ]
  }
]
```

そのあと以下を実行：

```bash
python model_creator.py
```

モデルは `ramen_model.h5` として保存されます。

---

## 🔁 追加学習（Fine-tuning）

追加データ（`qa_data.json`）を以下の形式で作成：

```json
[
  {
    "question": "味噌ラーメンの特徴は？",
    "answer": "濃厚な味噌だれが特徴です。"
  }
]
```

以下を実行：

```bash
python model_finetune.py
```

再学習後のモデルは `ramen_model_finetuned.h5` として保存されます。

---

## 🧪 推論（インファレンス）

推論用コード例（StreamlitやAPIに応用可能）：

```python
from transformers import BertTokenizer
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
model = tf.keras.models.load_model("ramen_model_finetuned.h5")

def predict_score(question, answer):
    inputs = tokenizer(question, answer, padding="max_length", max_length=64, truncation=True, return_tensors="tf")
    pred = model.predict([inputs["input_ids"], inputs["attention_mask"]])
    return float(pred[0][0])
```

---

## 🚼 .gitignore に入れておくと便利なもの

```
__pycache__/
*.pyc
*.h5
*.keras
.env
.vscode/
.idea/
```

---

## 📄 ライセンス

MIT License ✨
