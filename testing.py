from config.settings import settings

print("Anthropic:", bool(settings.ANTHROPIC_API_KEY))
print(settings.ANTHROPIC_API_KEY[:15])