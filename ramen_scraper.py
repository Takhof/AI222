import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Google検索を使ったページ取得
def google_search(query, num_results=5):
    print(links)
    headers = {"User-Agent": "Mozilla/5.0"}
    search_url = f"https://www.google.com/search?q={query}"
    res = requests.get(search_url, headers=headers)
    print(res.status_code)
    print(res.text[:1000])
    soup = BeautifulSoup(res.text, "html.parser")
    links = []
    for g in soup.select("a"):
        href = g.get("href")
        if href and "/url?q=" in href:
            url = href.split("/url?q=")[1].split("&")[0]
            if url.startswith("http"):
                links.append(url)
        if len(links) >= num_results:
            break
    return links

# スクレイピングで質問と答えを生成
def scrape_page_to_qa(url):
    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return None
        soup = BeautifulSoup(res.text, "html.parser")
        title = soup.title.string if soup.title else ""
        paragraph = soup.find("p").get_text(strip=True) if soup.find("p") else ""
        return {"question": title.strip(), "answer": paragraph.strip()}
    except Exception as e:
        print(f"⚠️ エラー: {url} -> {e}")
        return None

# 検索してデータを生成するフロー
def search_and_scrape(query):
    urls = google_search(query)
    results = []
    for url in urls:
        qa = scrape_page_to_qa(url)
        if qa:
            results.append(qa)
        time.sleep(1)  # 優しくしよう
    return pd.DataFrame(results)

df = search_and_scrape("ラーメン スープの作り方")
df.to_csv("qa_data.csv", mode="a", index=False, encoding="utf-8-sig")
print("✅ Google検索からデータ収集してCSVに保存したよ")
