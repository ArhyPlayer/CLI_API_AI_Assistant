import json
import os
from typing import Dict, List, Optional
from config import MAX_CONTEXT_LENGTH, DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS


class ContextManager:
    """
    Менеджер контекста для хранения истории диалогов пользователей.
    Использует словарь в памяти с опциональным сохранением в файл.
    """

    def __init__(self, storage_file: str = "user_contexts.json"):
        """
        Инициализация менеджера контекста.

        Args:
            storage_file: Путь к файлу для сохранения контекстов
        """
        self.storage_file = storage_file
        self.contexts: Dict[int, Dict] = {}  # user_id -> context_data

        # Загружаем сохраненные контексты при инициализации
        self._load_contexts()

    def get_context(self, user_id: int) -> Dict:
        """
        Получить контекст пользователя.

        Args:
            user_id: ID пользователя Telegram

        Returns:
            Словарь с данными контекста
        """
        if user_id not in self.contexts:
            # Создаем новый контекст для пользователя с дефолтными значениями
            self.contexts[user_id] = {
                "messages": [],
                "model": "gpt-3.5-turbo",
                "provider": "openai",
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "tokens_used": {
                    "openai": 0,
                    "anthropic": 0
                }
            }
            print(f"✨ Создан новый контекст для user {user_id} с system_prompt: \"{DEFAULT_SYSTEM_PROMPT}\"")

        return self.contexts[user_id]

    def update_context(self, user_id: int, messages: List[Dict], model: str = "gpt-3.5-turbo",
                      provider: str = "openai", system_prompt: Optional[str] = None,
                      temperature: float = 0.7, max_tokens: int = 1000) -> None:
        """
        Обновить контекст пользователя.

        Args:
            user_id: ID пользователя Telegram
            messages: Список сообщений
            model: Модель AI
            provider: Провайдер AI ("openai" или "anthropic")
            system_prompt: Системный промпт
        """
        # Ограничиваем длину контекста
        if len(messages) > MAX_CONTEXT_LENGTH:
            # Оставляем последние MAX_CONTEXT_LENGTH сообщений
            messages = messages[-MAX_CONTEXT_LENGTH:]

        # Сохраняем существующие токены, чтобы не потерять их при обновлении
        existing_tokens = {"openai": 0, "anthropic": 0}  # Инициализируем значениями по умолчанию
        if user_id in self.contexts and "tokens_used" in self.contexts[user_id]:
            existing_tokens = self.contexts[user_id]["tokens_used"]
            print(f"🔄 UPDATE_CONTEXT: Сохраняем существующие токены: {existing_tokens}")
        else:
            print(f"🔄 UPDATE_CONTEXT: Инициализируем новые токены для user {user_id}")

        self.contexts[user_id] = {
            "messages": messages,
            "model": model,
            "provider": provider,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tokens_used": existing_tokens  # Сохраняем токены!
        }

        print(f"🔄 UPDATE_CONTEXT: Обновлен контекст для user {user_id}, tokens_used: {existing_tokens}")

        # Сохраняем контексты в файл
        self._save_contexts()

    def clear_context(self, user_id: int) -> None:
        """
        Очистить контекст пользователя.

        Args:
            user_id: ID пользователя Telegram
        """
        if user_id in self.contexts:
            del self.contexts[user_id]
            self._save_contexts()

    def get_user_messages(self, user_id: int) -> List[Dict]:
        """
        Получить сообщения пользователя.

        Args:
            user_id: ID пользователя Telegram

        Returns:
            Список сообщений
        """
        context = self.get_context(user_id)
        return context.get("messages", [])

    def get_user_model(self, user_id: int) -> str:
        """
        Получить модель пользователя.

        Args:
            user_id: ID пользователя Telegram

        Returns:
            Название модели
        """
        context = self.get_context(user_id)
        return context.get("model", "gpt-3.5-turbo")

    def get_user_provider(self, user_id: int) -> str:
        """
        Получить провайдера пользователя.

        Args:
            user_id: ID пользователя Telegram

        Returns:
            Провайдер ("openai" или "anthropic")
        """
        context = self.get_context(user_id)
        return context.get("provider", "openai")

    def get_user_system_prompt(self, user_id: int) -> Optional[str]:
        """
        Получить системный промпт пользователя.

        Args:
            user_id: ID пользователя Telegram

        Returns:
            Системный промпт или None
        """
        context = self.get_context(user_id)
        return context.get("system_prompt")

    def get_user_temperature(self, user_id: int) -> float:
        """
        Получить температуру пользователя.

        Args:
            user_id: ID пользователя Telegram

        Returns:
            Температура (0.0-1.0)
        """
        context = self.get_context(user_id)
        return context.get("temperature", 0.7)

    def get_user_max_tokens(self, user_id: int) -> int:
        """
        Получить max_tokens пользователя.

        Args:
            user_id: ID пользователя Telegram

        Returns:
            Максимальное количество токенов
        """
        context = self.get_context(user_id)
        return context.get("max_tokens", 1000)

    def add_tokens_used(self, user_id: int, provider: str, tokens: int) -> None:
        """
        Добавить использованные токены к статистике.

        Args:
            user_id: ID пользователя Telegram
            provider: "openai" или "anthropic"
            tokens: Количество использованных токенов
        """
        print(f"🔍 ADD_TOKENS входные параметры: user_id={user_id}, provider={provider}, tokens={tokens} (type: {type(tokens)})")
        
        context = self.get_context(user_id)
        if "tokens_used" not in context:
            context["tokens_used"] = {"openai": 0, "anthropic": 0}
            print(f"🔍 ADD_TOKENS: Инициализирован tokens_used для user {user_id}")
        
        # Убеждаемся что структура tokens_used правильная
        if not isinstance(context["tokens_used"], dict):
            context["tokens_used"] = {"openai": 0, "anthropic": 0}
            print(f"⚠️ ADD_TOKENS: tokens_used не был словарем, переинициализирован")
        
        # Убеждаемся что provider существует в словаре
        if provider not in context["tokens_used"]:
            context["tokens_used"][provider] = 0
            print(f"⚠️ ADD_TOKENS: Добавлен отсутствующий provider '{provider}'")

        old_value = context["tokens_used"][provider]
        context["tokens_used"][provider] += tokens
        new_value = context["tokens_used"][provider]

        print(f"✅ ADD_TOKENS: User {user_id} - {provider} tokens: {old_value} + {tokens} = {new_value}")
        self._save_contexts()
        print(f"💾 ADD_TOKENS: Контекст сохранен на диск")

    def get_tokens_used(self, user_id: int, provider: str) -> int:
        """
        Получить количество использованных токенов для провайдера.

        Args:
            user_id: ID пользователя Telegram
            provider: "openai" или "anthropic"

        Returns:
            Количество использованных токенов
        """
        context = self.get_context(user_id)
        tokens_used_dict = context.get("tokens_used", {"openai": 0, "anthropic": 0})
        result = tokens_used_dict.get(provider, 0)
        print(f"🔍 GET_TOKENS: User {user_id}, provider {provider}, result: {result}, full dict: {tokens_used_dict}")
        return result

    def reset_tokens_used(self, user_id: int, provider: str) -> None:
        """
        Сбросить статистику токенов для провайдера.

        Args:
            user_id: ID пользователя Telegram
            provider: "openai" или "anthropic"
        """
        context = self.get_context(user_id)
        if "tokens_used" not in context:
            context["tokens_used"] = {"openai": 0, "anthropic": 0}

        context["tokens_used"][provider] = 0
        self._save_contexts()

    def _load_contexts(self) -> None:
        """Загрузить контексты из файла."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Конвертируем ключи обратно в int
                    self.contexts = {int(k): v for k, v in data.items()}
        except (json.JSONDecodeError, FileNotFoundError, ValueError):
            # Если файл поврежден или не существует, начинаем с пустого словаря
            self.contexts = {}

    def _save_contexts(self) -> None:
        """Сохранить контексты в файл."""
        try:
            print(f"💾 SAVE: Сохранение контекстов в {self.storage_file}")
            print(f"💾 SAVE: Данные: {json.dumps({k: {'tokens': v.get('tokens_used', {})} for k, v in self.contexts.items()}, ensure_ascii=False)}")
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.contexts, f, ensure_ascii=False, indent=2)
            print(f"✅ SAVE: Контексты успешно сохранены")
        except Exception as e:
            # В случае ошибки сохранения просто пропускаем
            print(f"❌ SAVE ERROR: {e}")
            pass
