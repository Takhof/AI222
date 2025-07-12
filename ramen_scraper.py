from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import json
import os
from dotenv import load_dotenv
import pandas as pd


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
print(f"🔑 OpenAI API Key (一部): {client}******")


# ChatGPTにURLリストをもらう
def get_ramen_urls_from_chatgpt(query="ラーメン スープの作り方を紹介している日本語のWebページを5つ教えて"):
    messages = [{"role": "user", "content": query}]
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.3
    )
    reply = response.choices[0].message.content
    print(reply)
    urls = [line.strip() for line in reply.splitlines() if line.strip().startswith("http")]
    return urls

# HTML本文を取得
def get_text_from_url(url):
    res = requests.get(url, timeout=10)
    soup = BeautifulSoup(res.text, "html.parser")
    texts = soup.find_all("p")
    return "\n".join(p.get_text(strip=True) for p in texts)

# ChatGPTに本文からQ&A作ってもらう
def extract_qa_from_text(text):
    prompt = f"""以下の文章から、ラーメンに関する質問と答えを5組、JSON形式で生成して下さい。

{text}

フォーマットは以下のようにしてください：
[
  {{
    "question": "ラーメンのスープはどうやって作りますか？",
    "answer": "豚骨や鶏ガラを長時間煮込んで作ります。"
  }},
  {{
    "question": "かん水とはなんですか？",
    "answer": "ラーメンの麺に使われるアルカリ性の液体で、独特の風味を与えます。"
  }}
]
"""
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.4
    )
    reply = response.choices[0].message.content
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        print("⚠️ JSONとして読み取れなかったよ")
        print(reply) 
        return []
    

# メイン処理
urls = get_ramen_urls_from_chatgpt()
all_qa = []
for url in urls:
    try:
        text = get_text_from_url(url)
        qa_pairs = extract_qa_from_text(text)
        all_qa.extend(qa_pairs)
    except Exception as e:
        print(f"❌ {url} でエラー: {e}")

# 保存
print(all_qa)
print(type(all_qa))
clean_qa = [qa for qa in all_qa if isinstance(qa, dict) and "question" in qa and "answer" in qa]

print(f"📦 保存対象Q&A数: {len(clean_qa)}")
if clean_qa:
    df = pd.DataFrame(clean_qa)
    df.to_csv("qa_data.csv", mode="a", index=False, header=not os.path.exists("qa_data.csv"), encoding="utf-8-sig")
    print("✅ ChatGPTを使ってラーメンQ&Aを自動収集・保存したよ♪")
else:
    print("⚠️ 有効なQ&Aが見つからなかったから、保存できなかったよ…")