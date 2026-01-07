import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Токены и настройки
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Проверяем наличие необходимых токенов
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не найден в переменных окружения. Добавьте BOT_TOKEN в файл .env")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не найден в переменных окружения. Добавьте OPENAI_API_KEY в файл .env")

# Настройки AI моделей
DEFAULT_MODEL_OPENAI = "gpt-3.5-turbo"
DEFAULT_MODEL_ANTHROPIC = "claude-sonnet-4-5-20250929"

# Настройки контекста
MAX_CONTEXT_LENGTH = 50  # Максимальное количество сообщений в контексте

# Настройки прокси API
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.proxyapi.ru/anthropic")

# Параметры генерации по умолчанию
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000
DEFAULT_SYSTEM_PROMPT = "По умолчанию."
