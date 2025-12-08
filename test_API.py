import os
from pprint import pprint

import anthropic
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Используем единый ключ из переменной окружения AI_API_KEY
api_key = os.getenv("AI_API_KEY")
if not api_key:
    raise ValueError("Не найден API ключ. Установите AI_API_KEY в .env")

client = anthropic.Anthropic(
    api_key=api_key,
    base_url="https://api.proxyapi.ru/anthropic",
)

message = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Привет!"}
            ],
        }
    ],
)

pprint(message)