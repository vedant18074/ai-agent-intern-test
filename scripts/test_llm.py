from app.llm import LLMService


llm = LLMService()

response = llm.generate(
    "Reply with exactly: LOCAL LLM SUCCESS"
)

print(response)