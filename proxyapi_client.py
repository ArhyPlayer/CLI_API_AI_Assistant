import os
from typing import List, Dict, Optional

import anthropic
from dotenv import load_dotenv
from openai import OpenAI

# Загружаем переменные окружения из .env файла
load_dotenv()


class ChatAI:
    """
    Класс для работы с OpenAI Chat Completions API и думающей моделью Anthropic.
    Поддерживает сохранение контекста разговора.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        provider: str = "openai",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_message: Optional[str] = None,
    ):
        """
        Инициализация чат-бота.

        Args:
            api_key: API ключ OpenAI/Anthropic. Если не указан, берется из переменных окружения.
            model: Модель для использования (по умолчанию gpt-3.5-turbo)
            provider: "openai" (обычная) или "anthropic" (думающая)
            temperature: Температура генерации (0.0 - 1.0)
            max_tokens: Максимальное количество токенов в ответе
            system_message: Системное сообщение (опционально)
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt: Optional[str] = system_message
        self.messages: List[Dict[str, str]] = []
        self.last_thinking_text: Optional[str] = None

        # Ключи
        openai_key = api_key or os.getenv("AI_API_KEY")
        anthropic_key = api_key or os.getenv("AI_API_KEY")

        # Клиенты
        if provider == "anthropic":
            if not anthropic_key:
                raise ValueError("API ключ Anthropic не найден. Установите AI_API_KEY или передайте api_key.")
            self.anthropic_client = anthropic.Anthropic(
                api_key=anthropic_key,
                base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.proxyapi.ru/anthropic"),
                timeout=60,
            )
            self.openai_client = None
        else:
            if not openai_key:
                raise ValueError("API ключ OpenAI не найден. Укажите его в конструкторе или установите AI_API_KEY.")
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

    def send_message(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        Отправляет сообщение и получает ответ от AI с сохранением контекста.

        Args:
            message: Сообщение пользователя
            system_prompt: Системный промпт (используется только при первом сообщении)

        Returns:
            Ответ от AI
        """
        # Устанавливаем системный промпт, если передан
        if system_prompt and not self.system_prompt:
            self.set_system_prompt(system_prompt)

        # Добавляем сообщение пользователя
        self.add_message("user", message)

        try:
            if self.provider == "anthropic":
                print("... отправляю запрос в Claude, подождите")
                print()  # Отступ после уведомления
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
                    
            else:
                response = self._send_openai()
                self.last_thinking_text = None
                ai_response = response.choices[0].message.content

            # Добавляем ответ AI в историю
            self.add_message("assistant", ai_response)

            # Выводим информацию о запросе
            self._print_request_info(response)

            return ai_response

        except Exception as e:
            import traceback
            error_msg = f"Ошибка при обращении к API ({self.provider}): {str(e)}"
            print(error_msg)
            print(f"Детали ошибки:\n{traceback.format_exc()}")
            # Удаляем последнее сообщение пользователя, так как запрос не выполнен
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            return error_msg

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

    def _print_request_info(self, response) -> None:
        """Выводит информацию о выполненном запросе."""
        try:
            if self.provider == "anthropic":
                # Для Anthropic получаем информацию об использованных токенах
                usage = getattr(response, 'usage', None)
                if usage:
                    tokens_used = getattr(usage, 'output_tokens', 0)
                else:
                    tokens_used = "неизвестно"
            else:
                # Для OpenAI получаем информацию об использованных токенах
                usage = getattr(response, 'usage', None)
                if usage:
                    tokens_used = getattr(usage, 'total_tokens', 0)
                else:
                    tokens_used = "неизвестно"

            print("\n" + "="*60)
            print("📊 ИНФОРМАЦИЯ О ЗАПРОСЕ:")
            print(f"   Модель: {self.model}")
            print(f"   Температура: {self.temperature}")
            print(f"   Max tokens: {self.max_tokens}")
            print(f"   Использовано токенов: {tokens_used}")
            print("="*60 + "\n")
        except Exception as e:
            # В случае ошибки выводим информацию без токенов
            print("\n" + "="*60)
            print("📊 ИНФОРМАЦИЯ О ЗАПРОСЕ:")
            print(f"   Модель: {self.model}")
            print(f"   Температура: {self.temperature}")
            print(f"   Max tokens: {self.max_tokens}")
            print("   Использовано токенов: неизвестно")
            print("="*60 + "\n")

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
            params["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        
        print(f"Отправка запроса с параметрами: model={params['model']}, max_tokens={params['max_tokens']}")
        print()  # Отступ после параметров
        return self.anthropic_client.messages.create(**params)


def main():
    """
    Пример использования ChatAI для демонстрации работы в режиме диалога.
    """
    try:
        print("Выберите режим работы:")
        print("1 - Обычная модель OpenAI (gpt-3.5-turbo) [по умолчанию]")
        print("2 - Думающая модель Anthropic (claude-sonnet-4-5-20250929)")
        mode = input("Введите 1 или 2 (Enter = 1): ").strip()

        if mode == "2":
            provider = "anthropic"
            model = "claude-sonnet-4-5-20250929"
            print("\nРежим: думающая модель Anthropic")
        else:
            provider = "openai"
            model = "gpt-3.5-turbo"
            print("\nРежим: обычная модель OpenAI")

        # Запрашиваем параметры у пользователя
        print("\nНастройка параметров для модели:")

        # Запрос для модели
        user_query = input("Введите запрос для модели: ").strip()
        if not user_query:
            user_query = "Привет! Расскажи о себе."

        # Температура
        while True:
            try:
                temp_input = input("Температура (0.0-1.0, по умолчанию 0.7): ").strip()
                temperature = float(temp_input) if temp_input else 0.7
                if 0.0 <= temperature <= 1.0:
                    break
                else:
                    print("Температура должна быть в диапазоне 0.0-1.0")
            except ValueError:
                print("Введите корректное число")

        # Max tokens
        while True:
            try:
                tokens_input = input("Max tokens (по умолчанию 1000): ").strip()
                max_tokens = int(tokens_input) if tokens_input else 1000
                if max_tokens > 0:
                    break
                else:
                    print("Max tokens должен быть положительным числом")
            except ValueError:
                print("Введите корректное число")

        # System message (опционально)
        system_message = input("System message (опционально, Enter для пропуска): ").strip()
        if not system_message:
            system_message = None

        chat = ChatAI(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_message=system_message
        )

        # Отправляем первый запрос
        print("\nОтправка первого запроса...")
        print()  # Пустая строка перед отправкой
        response = chat.send_message(user_query)

        if chat.provider == "anthropic" and chat.last_thinking_text:
            print()  # Пустая строка после получения ответа
            print(f"AI: Размышления: {chat.last_thinking_text}")
            print()  # Отступ после размышлений
            print(f"Ответ: {response}")
        else:
            print(f"AI: {response}")

        print()  # Пустая строка перед разделителем
        print("-" * 50)

        print("\nПродолжите диалог с ИИ.")
        print("Команды:")
        print("  'exit'           - выход")
        print("  'switch claude'  - переключиться на Claude")
        print("  'switch openai'  - переключиться на OpenAI")
        print("  'model info'     - информация о текущей модели")
        print("  'clear'          - очистить историю")
        print("-" * 50)
        print()  # Отступ перед началом диалога

        while True:
            user_input = input("Вы: ").strip()
            lower_input = user_input.lower()

            if lower_input == "exit":
                break
            if lower_input == "switch claude":
                provider = "anthropic"
                model = "claude-sonnet-4-5-20250929"
                print("\nПереключение на Claude. Настройка параметров:")

                # Запрос для модели
                user_query = input("Введите запрос для модели: ").strip()
                if not user_query:
                    user_query = "Привет! Расскажи о себе."

                # Температура
                while True:
                    try:
                        temp_input = input("Температура (0.0-1.0, по умолчанию 0.7): ").strip()
                        temperature = float(temp_input) if temp_input else 0.7
                        if 0.0 <= temperature <= 1.0:
                            break
                        else:
                            print("Температура должна быть в диапазоне 0.0-1.0")
                    except ValueError:
                        print("Введите корректное число")

                # Max tokens
                while True:
                    try:
                        tokens_input = input("Max tokens (по умолчанию 1000): ").strip()
                        max_tokens = int(tokens_input) if tokens_input else 1000
                        if max_tokens > 0:
                            break
                        else:
                            print("Max tokens должен быть положительным числом")
                    except ValueError:
                        print("Введите корректное число")

                # System message (опционально)
                system_message = input("System message (опционально, Enter для пропуска): ").strip()
                if not system_message:
                    system_message = None

                chat = ChatAI(
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_message=system_message
                )
                print("\nПереключено на Claude (claude-sonnet-4-5-20250929). История сброшена.")
                print("-" * 50)
                print()

                # Отправляем первый запрос
                print()  # Пустая строка перед отправкой
                response = chat.send_message(user_query)

                if chat.provider == "anthropic" and chat.last_thinking_text:
                    print()  # Пустая строка после получения ответа
                    print(f"AI: Размышления: {chat.last_thinking_text}")
                    print()  # Отступ после размышлений
                    print(f"Ответ: {response}")
                else:
                    print(f"AI: {response}")

                print()  # Пустая строка перед разделителем
                print("-" * 50)
                continue
            if lower_input == "switch openai":
                provider = "openai"
                model = "gpt-3.5-turbo"
                print("\nПереключение на OpenAI. Настройка параметров:")

                # Запрос для модели
                user_query = input("Введите запрос для модели: ").strip()
                if not user_query:
                    user_query = "Привет! Расскажи о себе."

                # Температура
                while True:
                    try:
                        temp_input = input("Температура (0.0-1.0, по умолчанию 0.7): ").strip()
                        temperature = float(temp_input) if temp_input else 0.7
                        if 0.0 <= temperature <= 1.0:
                            break
                        else:
                            print("Температура должна быть в диапазоне 0.0-1.0")
                    except ValueError:
                        print("Введите корректное число")

                # Max tokens
                while True:
                    try:
                        tokens_input = input("Max tokens (по умолчанию 1000): ").strip()
                        max_tokens = int(tokens_input) if tokens_input else 1000
                        if max_tokens > 0:
                            break
                        else:
                            print("Max tokens должен быть положительным числом")
                    except ValueError:
                        print("Введите корректное число")

                # System message (опционально)
                system_message = input("System message (опционально, Enter для пропуска): ").strip()
                if not system_message:
                    system_message = None

                chat = ChatAI(
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_message=system_message
                )
                print("\nПереключено на OpenAI (gpt-3.5-turbo). История сброшена.")
                print("-" * 50)
                print()

                # Отправляем первый запрос
                print()  # Пустая строка перед отправкой
                response = chat.send_message(user_query)

                if chat.provider == "anthropic" and chat.last_thinking_text:
                    print()  # Пустая строка после получения ответа
                    print(f"AI: Размышления: {chat.last_thinking_text}")
                    print()  # Отступ после размышлений
                    print(f"Ответ: {response}")
                else:
                    print(f"AI: {response}")

                print()  # Пустая строка перед разделителем
                print("-" * 50)
                continue
            if lower_input == "model info":
                print(f"\nТекущий провайдер: {provider}, модель: {model}")
                print("-" * 50)
                print()
                continue
            if lower_input == "clear":
                chat.clear_history()
                print("\nИстория очищена.")
                print("-" * 50)
                print()
                continue

            # Отправляем обычное сообщение
            print()  # Пустая строка перед отправкой
            response = chat.send_message(user_input)
            
            if chat.provider == "anthropic" and chat.last_thinking_text:
                print()  # Пустая строка после получения ответа
                print(f"AI: Размышления: {chat.last_thinking_text}")
                print()  # Отступ после размышлений
                print(f"Ответ: {response}")
            else:
                print(f"AI: {response}")
            
            print()  # Пустая строка перед разделителем
            print("-" * 50)

    except ValueError as e:
        print(f"Ошибка инициализации: {e}")
        print("Установите переменную окружения AI_API_KEY или передайте api_key в конструктор ChatAI")


if __name__ == "__main__":
    main()
