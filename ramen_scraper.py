from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import json
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
print(f"🔑 OpenAI API Key (一部): {client}******")


# ChatGPTにURLリストをもらう
def get_ramen_urls_from_chatgpt(query="ラーメン スープの作り方を紹介している日本語のWebページを5つ教えて"):
    messages = [{"role": "user", "content": query}]
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.3
    )
    print("🗨️ GPT-4 からの返信内容:")
    print(response['choices'][0]['message']['content'])
    reply = response['choices'][0]['message']['content']
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
        model="gpt-4",
        messages=messages,
        temperature=0.4
    )
    try:
        return json.loads(response['choices'][0]['message']['content'])
    except json.JSONDecodeError:
        print("⚠️ JSONとして読み取れなかったよ")
        print(response['choices'][0]['message']['content']) 
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
import pandas as pd
df = pd.DataFrame(all_qa)
df.to_csv("qa_data.csv", mode="a", index=False, encoding="utf-8-sig")
print("✅ ChatGPTを使ってラーメンQ&Aを自動収集・保存したよ♪")