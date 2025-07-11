import json


with open("retriever_qa.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# 学習用ペアをつくる（質問, 答え, ラベル）形式
train_pairs = []

for item in qa_data:
    q = item["question"]
    train_pairs.append((q, item["positive"], 1))  # 正解ペア
    for neg in item["negatives"]:
        train_pairs.append((q, neg, 0))  # 間違いペア

# 例として10件表示
for i in range(10):
    print(train_pairs[i])