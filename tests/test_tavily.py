from tools.tavily_search import TavilySearch

tool = TavilySearch()

results = tool.search("Artificial Intelligence")

print(results)