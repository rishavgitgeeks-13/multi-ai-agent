from tools.news_search import NewsSearch

news = NewsSearch()

results = news.search(
    "OpenAI GPT-5"
)

print(
    f"Results: {len(results)}"
)

for i, item in enumerate(results, start=1):
    print("\n" + "=" * 80)
    print(f"Result {i}")
    print("Title:", item.get("title"))
    print("Source:", item.get("source"))
    print("Published:", item.get("published_at"))
    print("URL:", item.get("url"))
    print(
        "Content Length:",
        len(
            item.get("content")
            or item.get("description")
            or ""
        )
    )