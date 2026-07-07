from models.llm import llm

response = llm.invoke(
    "Explain AI Agents in one paragraph."
)

print(response.content)