import os
from typing import List, Dict, Optional

import anthropic
from dotenv import load_dotenv
from openai import OpenAI

# Загружаем переменные окружения из .env файла
load_dotenv()


class ProxyAPIClient:
    """
    Класс для работы с OpenAI Chat Completions API и думающей моделью Anthropic через proxyapi.
    Поддерживает сохранение контекста разговора.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Инициализация клиента для работы с proxyapi.

        Args:
            api_key: API ключ OpenAI/Anthropic. Если не указан, берется из переменных окружения.
            model: Модель для использования (по умолчанию gpt-3.5-turbo)
            provider: "openai" (обычная) или "anthropic" (думающая)
            temperature: Температура генерации (0.0-1.0)
            max_tokens: Максимальное количество токенов в ответе
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt: Optional[str] = None
        self.messages: List[Dict[str, str]] = []
        self.last_thinking_text: Optional[str] = None

        # Ключи
        openai_key = api_key or os.getenv("OPENAI_API_KEY")
        anthropic_key = api_key or os.getenv("OPENAI_API_KEY")

        # Клиенты
        if provider == "anthropic":
            if not anthropic_key:
                raise ValueError("API ключ Anthropic не найден. Установите OPENAI_API_KEY или передайте api_key.")
            self.anthropic_client = anthropic.Anthropic(
                api_key=anthropic_key,
                base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.proxyapi.ru/anthropic"),
                timeout=60,
            )
            self.openai_client = None
        else:
            if not openai_key:
                raise ValueError("API ключ OpenAI не найден. Укажите его в конструкторе или установите OPENAI_API_KEY.")
            self.openai_client = OpenAI(
                api_key=openai_key,
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1"),
            )
            self.anthropic_client = None

    def add_message(self, role: str, content: str) -> None:
        """
        Добавляет сообщение в историю чата.

        Args:
            role: Роль отправителя ('user', 'assistant', 'system')
            content: Текст сообщения
        """
        self.messages.append({"role": role, "content": content})

    def send_message(self, message: str, system_prompt: Optional[str] = None) -> tuple[str, int]:
        """
        Отправляет сообщение и получает ответ от AI с сохранением контекста.

        Args:
            message: Сообщение пользователя
            system_prompt: Системный промпт (используется только при первом сообщении)

        Returns:
            Кортеж (ответ от AI, количество использованных токенов)
        """
        # Устанавливаем системный промпт, если передан
        if system_prompt and not self.system_prompt:
            self.set_system_prompt(system_prompt)

        # Добавляем сообщение пользователя
        self.add_message("user", message)

        try:
            if self.provider == "anthropic":
                response = self._send_anthropic()

                # Извлекаем размышления и текстовый ответ
                thinking_blocks = []
                text_blocks = []

                for block in response.content:
                    if hasattr(block, 'type'):
                        if block.type == "thinking":
                            # У ThinkingBlock атрибут называется 'thinking', а не 'text'
                            thinking_blocks.append(block.thinking)
                        elif block.type == "text":
                            text_blocks.append(block.text)

                self.last_thinking_text = "\n".join(thinking_blocks) if thinking_blocks else None
                ai_response = "".join(text_blocks)

                if not ai_response:
                    ai_response = "⚠️ Получен пустой ответ от Claude"

                # Получаем количество использованных токенов для Anthropic
                print(f"DEBUG: Anthropic response type: {type(response)}")
                print(f"DEBUG: Anthropic response dir: {[attr for attr in dir(response) if not attr.startswith('_')]}")
                usage = getattr(response, 'usage', None)
                print(f"DEBUG: Anthropic usage object: {usage}")
                tokens_used = 0
                if usage:
                    print(f"DEBUG: Anthropic usage dir: {[attr for attr in dir(usage) if not attr.startswith('_')]}")
                    # Пробуем разные поля для токенов
                    output_tokens = getattr(usage, 'output_tokens', 0)
                    total_tokens = getattr(usage, 'total_tokens', 0)
                    input_tokens = getattr(usage, 'input_tokens', 0)
                    print(f"DEBUG: Anthropic usage fields - output: {output_tokens}, total: {total_tokens}, input: {input_tokens}")

                    tokens_used = output_tokens or total_tokens or (input_tokens + output_tokens)
                    print(f"DEBUG: Anthropic calculated tokens: {tokens_used}")

                # Всегда используем оценку, если токены не найдены из API
                if tokens_used == 0 and ai_response:
                    # Более точная оценка для Anthropic (обычно 1 токен = ~4 символа)
                    tokens_used = max(1, len(ai_response) // 4)
                    print(f"DEBUG: Anthropic tokens estimated: {tokens_used} (from {len(ai_response)} chars)")
                elif tokens_used == 0:
                    # Если даже ответа нет, используем минимальное значение
                    tokens_used = 1
                    print(f"DEBUG: Anthropic tokens set to minimum: {tokens_used}")

            else:
                response = self._send_openai()
                self.last_thinking_text = None
                ai_response = response.choices[0].message.content

                # Получаем количество использованных токенов для OpenAI
                print(f"DEBUG: OpenAI response type: {type(response)}")
                print(f"DEBUG: OpenAI response dir: {[attr for attr in dir(response) if not attr.startswith('_')]}")
                usage = getattr(response, 'usage', None)
                print(f"DEBUG: OpenAI usage object: {usage}")
                tokens_used = 0
                if usage:
                    print(f"DEBUG: OpenAI usage dir: {[attr for attr in dir(usage) if not attr.startswith('_')]}")
                    total_tokens = getattr(usage, 'total_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0)
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    print(f"DEBUG: OpenAI usage fields - total: {total_tokens}, completion: {completion_tokens}, prompt: {prompt_tokens}")

                    tokens_used = total_tokens or completion_tokens or (prompt_tokens + completion_tokens)
                    print(f"DEBUG: OpenAI calculated tokens: {tokens_used}")

                # Всегда используем оценку, если токены не найдены из API
                if tokens_used == 0 and ai_response:
                    # OpenAI токены: примерно 1 токен = 0.75 слова или 4 символа
                    tokens_used = max(1, len(ai_response) // 4)
                    print(f"DEBUG: OpenAI tokens estimated: {tokens_used} (from {len(ai_response)} chars)")
                elif tokens_used == 0:
                    # Если даже ответа нет, используем минимальное значение
                    tokens_used = 1
                    print(f"DEBUG: OpenAI tokens set to minimum: {tokens_used}")

            # Добавляем ответ AI в историю
            self.add_message("assistant", ai_response)

            # Гарантируем что tokens_used это int и минимум 1
            tokens_used = max(1, int(tokens_used))
            print(f"🔍 FINAL: Returning tokens_used = {tokens_used} (type: {type(tokens_used)})")

            return ai_response, tokens_used

        except Exception as e:
            error_msg = f"Ошибка при обращении к API ({self.provider}): {str(e)}"
            # Удаляем последнее сообщение пользователя, так как запрос не выполнен
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            return error_msg, 0

    def clear_history(self) -> None:
        """Очищает историю сообщений."""
        self.messages = []

    def get_history(self) -> List[Dict[str, str]]:
        """
        Возвращает историю сообщений.

        Returns:
            Список сообщений в формате [{"role": "user", "content": "text"}, ...]
        """
        return self.messages.copy()

    def set_system_prompt(self, prompt: str) -> None:
        """
        Устанавливает системный промпт. Заменяет предыдущий системный промпт или добавляет новый.

        Args:
            prompt: Текст системного промпта
        """
        # Сохраняем системный промпт и убираем из истории системные сообщения
        self.system_prompt = prompt
        self.messages = [msg for msg in self.messages if msg["role"] != "system"]

    def _openai_messages(self) -> List[Dict[str, str]]:
        """Готовим сообщения в формате OpenAI."""
        msgs = self.messages.copy()
        if self.system_prompt:
            msgs = [{"role": "system", "content": self.system_prompt}] + msgs
        return msgs

    def _send_openai(self):
        """Запрос к OpenAI."""
        if not self.openai_client:
            raise ValueError("Клиент OpenAI не инициализирован.")

        return self.openai_client.chat.completions.create(
            model=self.model,
            messages=self._openai_messages(),
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
        )

    def _anthropic_messages(self) -> List[Dict[str, object]]:
        """Конвертация истории в формат Anthropic."""
        converted = []
        for msg in self.messages:
            if msg["role"] == "system":
                continue
            if msg["role"] not in ("user", "assistant"):
                continue
            converted.append(
                {
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}],
                }
            )
        return converted

    def _send_anthropic(self):
        """Запрос к Anthropic (думающая модель)."""
        if not self.anthropic_client:
            raise ValueError("Клиент Anthropic не инициализирован.")

        # Для моделей с расширенным мышлением включаем thinking
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._anthropic_messages(),
        }

        # Добавляем системный промпт, если он есть
        if self.system_prompt:
            params["system"] = self.system_prompt

        # Для Sonnet 4.5 включаем extended thinking
        if "sonnet-4-5" in self.model or "sonnet-4.5" in self.model:
            # Убеждаемся что max_tokens достаточно для thinking
            # budget_tokens должен быть меньше max_tokens
            # Оставляем минимум 512 токенов для ответа
            if self.max_tokens < 1536:  # Минимум для thinking (1024) + ответ (512)
                # Увеличиваем max_tokens до минимально необходимого
                params["max_tokens"] = 2048
                budget_tokens = 1024
                print(f"⚠️ ANTHROPIC: max_tokens увеличен до {params['max_tokens']} для extended thinking")
            else:
                # Используем 2/3 от max_tokens для thinking, 1/3 для ответа
                budget_tokens = min(1024, int(self.max_tokens * 0.66))
                print(f"ℹ️ ANTHROPIC: budget_tokens установлен в {budget_tokens} (max_tokens: {self.max_tokens})")
            
            params["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}

        return self.anthropic_client.messages.create(**params)
