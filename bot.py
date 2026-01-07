import asyncio
import logging
import time
from typing import Optional

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton

from config import BOT_TOKEN, DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from context_manager import ContextManager
from proxyapi_client import ProxyAPIClient

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Функции для создания клавиатур
def get_main_keyboard() -> InlineKeyboardMarkup:
    """Создает основную клавиатуру с командами."""
    keyboard = [
        [
            InlineKeyboardButton(text="🤖 Модель: GPT", callback_data="switch_openai"),
            InlineKeyboardButton(text="🧠 Модель: Claude", callback_data="switch_claude"),
        ],
        [
            InlineKeyboardButton(text="⚙️ Настройки", callback_data="show_settings"),
            InlineKeyboardButton(text="📊 Статистика", callback_data="show_stats"),
        ],
        [
            InlineKeyboardButton(text="🧹 Очистить контекст", callback_data="clear_context"),
            InlineKeyboardButton(text="❓ Помощь", callback_data="show_help"),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def get_model_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру выбора модели."""
    keyboard = [
        [
            InlineKeyboardButton(text="🤖 GPT-3.5-turbo", callback_data="switch_openai"),
            InlineKeyboardButton(text="🧠 Claude Sonnet", callback_data="switch_claude"),
        ],
        [
            InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_main"),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def get_help_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру помощи."""
    keyboard = [
        [
            InlineKeyboardButton(text="🎯 Основные команды", callback_data="show_commands"),
        ],
        [
            InlineKeyboardButton(text="📚 О моделях", callback_data="show_models_info"),
        ],
        [
            InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_main"),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def get_settings_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру настроек."""
    keyboard = [
        [
            InlineKeyboardButton(text="🌡️ Температура", callback_data="set_temperature"),
            InlineKeyboardButton(text="📏 Max Tokens", callback_data="set_max_tokens"),
        ],
        [
            InlineKeyboardButton(text="💬 System Message", callback_data="set_system_message"),
            InlineKeyboardButton(text="🔄 Сбросить", callback_data="reset_settings"),
        ],
        [
            InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_main"),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def get_temperature_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру выбора температуры."""
    keyboard = [
        [
            InlineKeyboardButton(text="🎯 0.0 (точный)", callback_data="temp_0.0"),
            InlineKeyboardButton(text="⚖️ 0.7 (сбалансированный)", callback_data="temp_0.7"),
        ],
        [
            InlineKeyboardButton(text="🎨 1.0 (творческий)", callback_data="temp_1.0"),
            InlineKeyboardButton(text="🔥 1.5 (экспериментальный)", callback_data="temp_1.5"),
        ],
        [
            InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_settings"),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def get_max_tokens_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру выбора max_tokens."""
    keyboard = [
        [
            InlineKeyboardButton(text="💬 500 (короткий)", callback_data="tokens_500"),
            InlineKeyboardButton(text="📝 1000 (стандарт)", callback_data="tokens_1000"),
        ],
        [
            InlineKeyboardButton(text="📚 2000 (длинный)", callback_data="tokens_2000"),
            InlineKeyboardButton(text="🎯 4000 (максимальный)", callback_data="tokens_4000"),
        ],
        [
            InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_settings"),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def get_menu_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру с кнопкой меню."""
    keyboard = [
        [
            InlineKeyboardButton(text="📱 Меню", callback_data="back_to_main"),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

def set_user_state(user_id: int, state: str) -> None:
    """Установить состояние пользователя с timestamp."""
    user_states[user_id] = {
        "state": state,
        "timestamp": time.time()
    }
    logger.info(f"Установлено состояние {state} для пользователя {user_id}")

def get_user_state(user_id: int) -> Optional[str]:
    """Получить состояние пользователя с проверкой таймаута."""
    if user_id not in user_states:
        return None

    state_data = user_states[user_id]
    if time.time() - state_data["timestamp"] > STATE_TIMEOUT:
        # Состояние истекло
        del user_states[user_id]
        logger.info(f"Состояние {state_data['state']} истекло для пользователя {user_id}")
        return None

    return state_data["state"]

def clear_user_state(user_id: int) -> None:
    """Очистить состояние пользователя."""
    if user_id in user_states:
        del user_states[user_id]
        logger.info(f"Состояние очищено для пользователя {user_id}")

def get_back_keyboard() -> InlineKeyboardMarkup:
    """Создает клавиатуру с кнопкой назад."""
    keyboard = [
        [
            InlineKeyboardButton(text="⬅️ Назад", callback_data="back_to_main"),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)

# Константы
STATE_TIMEOUT = 300  # 5 минут таймаут для состояний

# Инициализация бота и диспетчера
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Инициализация менеджера контекстов
context_manager = ContextManager()

# Словарь для отслеживания состояний пользователей
user_states = {}  # user_id -> {"state": state, "timestamp": timestamp}


def get_ai_client(user_id: int) -> ProxyAPIClient:
    """
    Создать или получить AI клиент для пользователя.

    Args:
        user_id: ID пользователя Telegram

    Returns:
        Экземпляр ProxyAPIClient
    """
    # Получаем настройки пользователя из контекста
    model = context_manager.get_user_model(user_id)
    provider = context_manager.get_user_provider(user_id)
    system_prompt = context_manager.get_user_system_prompt(user_id)
    temperature = context_manager.get_user_temperature(user_id)
    max_tokens = context_manager.get_user_max_tokens(user_id)

    # Создаем клиент с настройками пользователя
    client = ProxyAPIClient(
        model=model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Устанавливаем системный промпт, если есть
    if system_prompt:
        client.set_system_prompt(system_prompt)
    elif not system_prompt:
        # Устанавливаем дефолтный системный промпт
        client.set_system_prompt(DEFAULT_SYSTEM_PROMPT)

    # Загружаем историю сообщений
    messages = context_manager.get_user_messages(user_id)
    client.messages = messages.copy()

    return client


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    """
    Обработчик команды /start.
    """
    user_id = message.from_user.id
    username = message.from_user.username or "пользователь"

    logger.info(f"Пользователь {user_id} ({username}) запустил бота")

    welcome_text = (
        f"Привет, {username}! 👋\n\n"
        "Я - AI помощник с поддержкой нескольких моделей:\n"
        "• 🤖 GPT-3.5-turbo\n"
        "• 🧠 Claude Sonnet 4.5\n\n"
        "Выберите действие или просто напишите сообщение:"
    )

    # Получаем текущую модель пользователя
    current_model = context_manager.get_user_model(user_id)
    current_provider = context_manager.get_user_provider(user_id)

    model_name = "GPT-3.5-turbo" if current_provider == "openai" else "Claude Sonnet"
    welcome_text += f"\n\n🎯 Текущая модель: {model_name}"

    await message.reply(welcome_text, reply_markup=get_main_keyboard())


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """
    Обработчик команды /help.
    """
    help_text = (
        "🤖 AI Чат-бот\n\n"
        "📝 Просто отправьте сообщение - бот ответит с учетом контекста.\n\n"
        "🎮 Или используйте удобные кнопки ниже:"
    )

    await message.reply(help_text, reply_markup=get_main_keyboard())


@dp.message(Command("switch_openai"))
async def cmd_switch_openai(message: Message) -> None:
    """
    Переключение на модель OpenAI GPT-3.5-turbo.
    """
    user_id = message.from_user.id

    # Сохраняем текущий контекст перед сменой модели
    client = get_ai_client(user_id)
    context_manager.update_context(
        user_id=user_id,
        messages=client.messages,
        model="gpt-3.5-turbo",
        provider="openai",
        system_prompt=client.system_prompt,
        temperature=client.temperature,
        max_tokens=client.max_tokens
    )

    await message.reply("✅ Переключено на 🤖 GPT-3.5-turbo\nИстория сохранена.", reply_markup=get_main_keyboard())


@dp.message(Command("switch_claude"))
async def cmd_switch_claude(message: Message) -> None:
    """
    Переключение на модель Anthropic Claude.
    """
    user_id = message.from_user.id

    # Сохраняем текущий контекст перед сменой модели
    client = get_ai_client(user_id)
    context_manager.update_context(
        user_id=user_id,
        messages=client.messages,
        model="claude-sonnet-4-5-20250929",
        provider="anthropic",
        system_prompt=client.system_prompt,
        temperature=client.temperature,
        max_tokens=client.max_tokens
    )

    await message.reply("✅ Переключено на 🧠 Claude Sonnet 4.5\nИстория сохранена.", reply_markup=get_main_keyboard())


@dp.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    """
    Показать статистику использования токенов.
    """
    user_id = message.from_user.id

    openai_tokens = context_manager.get_tokens_used(user_id, "openai")
    anthropic_tokens = context_manager.get_tokens_used(user_id, "anthropic")
    total_tokens = openai_tokens + anthropic_tokens

    current_provider = context_manager.get_user_provider(user_id)
    current_model_name = "GPT-3.5-turbo" if current_provider == "openai" else "Claude Sonnet"

    stats_text = (
        f"📊 Статистика использования токенов:\n\n"
        f"🤖 GPT-3.5-turbo: {openai_tokens:,} токенов\n"
        f"🧠 Claude Sonnet: {anthropic_tokens:,} токенов\n"
        f"📈 Всего: {total_tokens:,} токенов\n\n"
        f"🎯 Текущая модель: {current_model_name}"
    )

    await message.reply(stats_text, reply_markup=get_main_keyboard())


@dp.message(Command("reset_stats"))
async def cmd_reset_stats(message: Message) -> None:
    """
    Сбросить всю статистику токенов.
    """
    user_id = message.from_user.id

    # Сбрасываем статистику для обеих моделей
    context_manager.reset_tokens_used(user_id, "openai")
    context_manager.reset_tokens_used(user_id, "anthropic")

    logger.info(f"Пользователь {user_id} сбросил всю статистику токенов")

    await message.reply("🧹 Вся статистика токенов сброшена!\n\nМожно начинать заново отслеживать использование.", reply_markup=get_main_keyboard())


@dp.message(Command("status"))
async def cmd_status(message: Message) -> None:
    """
    Показать текущий статус пользователя (для отладки).
    """
    user_id = message.from_user.id

    user_state = user_states.get(user_id, "normal")
    current_model = context_manager.get_user_model(user_id)
    current_provider = context_manager.get_user_provider(user_id)
    current_temp = context_manager.get_user_temperature(user_id)
    current_tokens = context_manager.get_user_max_tokens(user_id)
    system_prompt = context_manager.get_user_system_prompt(user_id)
    messages_count = len(context_manager.get_user_messages(user_id))

    # Форматируем отображение system prompt
    if system_prompt:
        # Показываем первые 100 символов system prompt
        system_display = system_prompt[:100]
        if len(system_prompt) > 100:
            system_display += "..."
        system_status = f"\"{system_display}\""
    else:
        system_status = "По умолчанию"

    status_text = (
        f"🔍 **Статус пользователя:**\n\n"
        f"👤 ID: {user_id}\n"
        f"📊 Состояние: {user_state}\n"
        f"🎯 Модель: {current_model} ({current_provider})\n"
        f"🌡️ Температура: {current_temp}\n"
        f"📏 Max tokens: {current_tokens}\n"
        f"💬 System message: {system_status}\n"
        f"💭 Сообщений в истории: {messages_count}\n"
    )

    await message.reply(status_text, reply_markup=get_main_keyboard())


@dp.message(Command("clear"))
async def cmd_clear(message: Message) -> None:
    """
    Очистка контекста разговора.
    """
    user_id = message.from_user.id

    # Получаем текущего провайдера и очищаем контекст
    current_provider = context_manager.get_user_provider(user_id)
    context_manager.clear_context(user_id)

    # Сбрасываем статистику для текущей модели
    context_manager.reset_tokens_used(user_id, current_provider)

    model_name = "GPT" if current_provider == "openai" else "Claude"
    logger.info(f"Пользователь {user_id} очистил контекст и статистику {current_provider}")

    await message.reply(
        f"🧹 Контекст очищен!\n📊 Статистика {model_name} сброшена!\n\nНачнем разговор заново.",
        reply_markup=get_main_keyboard()
    )


@dp.message()
async def handle_message(message: Message) -> None:
    """
    Обработчик обычных сообщений пользователя.
    """
    user_id = message.from_user.id
    user_text = message.text.strip() if message.text else ""

    if not user_text:
        return

    # Игнорируем команды (они обрабатываются отдельными обработчиками)
    if user_text.startswith('/'):
        return

    # Проверяем специальную команду "очистить контекст"
    if user_text.lower() == "очистить контекст":
        await cmd_clear(message)
        return

    # Проверяем состояние пользователя
    user_state = get_user_state(user_id)
    logger.info(f"Пользователь {user_id} состояние: {user_state}, сообщение: '{user_text[:50]}...'")

    if user_state == "waiting_system_message":
        logger.info(f"Обработка system message для пользователя {user_id}: '{user_text}'")

        # Пользователь устанавливает системное сообщение
        if user_text == "-":
            # Сброс к значению по умолчанию
            system_message = DEFAULT_SYSTEM_PROMPT
            response_text = f"✅ System message сброшен к значению по умолчанию:\n\"{DEFAULT_SYSTEM_PROMPT}\""
        else:
            # Установка нового системного сообщения
            system_message = user_text
            response_text = f"✅ System message установлен:\n\"{user_text}\""

        try:
            # Обновляем контекст
            client = get_ai_client(user_id)
            context_manager.update_context(
                user_id=user_id,
                messages=client.messages,
                model=client.model,
                provider=client.provider,
                system_prompt=system_message,
                temperature=client.temperature,
                max_tokens=client.max_tokens
            )
            logger.info(f"Контекст обновлен для пользователя {user_id} с system_prompt: {system_message}")
        except Exception as e:
            logger.error(f"Ошибка при обновлении контекста для system message: {e}")

        # Сбрасываем состояние пользователя
        clear_user_state(user_id)

        await message.reply(response_text, reply_markup=get_settings_keyboard())
        return

    logger.info(f"Получено сообщение от пользователя {user_id}: {user_text[:50]}...")

    try:
        # Показываем индикатор "печатает"
        await bot.send_chat_action(message.chat.id, "typing")

        # Получаем AI клиент для пользователя
        client = get_ai_client(user_id)

        # Отправляем запрос к AI
        response, tokens_used = client.send_message(user_text)

        # Сохраняем статистику токенов
        logger.info(f"🔍 DEBUG: Токены от API - тип: {type(tokens_used)}, значение: {tokens_used}")
        logger.info(f"🔍 DEBUG: Провайдер: {client.provider}")
        
        # Проверяем что tokens_used это число
        if isinstance(tokens_used, str):
            try:
                tokens_used = int(tokens_used)
                logger.info(f"🔍 DEBUG: Конвертировали tokens из строки в int: {tokens_used}")
            except ValueError:
                logger.error(f"❌ Не удалось конвертировать tokens '{tokens_used}' в int")
                tokens_used = 0
        
        logger.info(f"Токены использованы: {tokens_used} для провайдера {client.provider}")
        if tokens_used > 0:
            context_manager.add_tokens_used(user_id, client.provider, tokens_used)
            logger.info(f"Статистика обновлена: +{tokens_used} токенов для {client.provider}")
        else:
            logger.warning(f"⚠️ Токены равны 0, статистика не обновлена!")

        # Сохраняем обновленный контекст
        context_manager.update_context(
            user_id=user_id,
            messages=client.messages,
            model=client.model,
            provider=client.provider,
            system_prompt=client.system_prompt,
            temperature=client.temperature,
            max_tokens=client.max_tokens
        )

        # Отправляем ответ пользователю
        await message.reply(response, parse_mode="Markdown", reply_markup=get_menu_keyboard())

        # Если есть размышления Claude, отправляем их отдельно
        if client.provider == "anthropic" and client.last_thinking_text:
            thinking_message = f"🤔 *Размышления:*\n```\n{client.last_thinking_text}\n```"
            await message.reply(thinking_message, parse_mode="Markdown")

        logger.info(f"Отправлен ответ пользователю {user_id}")

    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения пользователя {user_id}: {str(e)}", exc_info=True)
        error_message = (
            "❌ Произошла ошибка при обработке вашего запроса.\n"
            "Попробуйте еще раз или обратитесь к администратору."
        )
        await message.reply(error_message)


# Обработчики callback-запросов (инлайн кнопки)
@dp.callback_query()
async def handle_callback(callback: CallbackQuery) -> None:
    """
    Обработчик callback-запросов от инлайн кнопок.
    """
    user_id = callback.from_user.id
    callback_data = callback.data

    logger.info(f"Callback от пользователя {user_id}: {callback_data}")

    try:
        if callback_data == "switch_openai":
            # Переключение на OpenAI GPT-3.5-turbo
            client = get_ai_client(user_id)
            context_manager.update_context(
                user_id=user_id,
                messages=client.messages,
                model="gpt-3.5-turbo",
                provider="openai",
                system_prompt=client.system_prompt,
                temperature=client.temperature,
                max_tokens=client.max_tokens
            )

            await callback.message.edit_text(
                "✅ Переключено на 🤖 GPT-3.5-turbo\n\nИстория сохранена.",
                reply_markup=get_main_keyboard()
            )

        elif callback_data == "switch_claude":
            # Переключение на Anthropic Claude
            client = get_ai_client(user_id)
            context_manager.update_context(
                user_id=user_id,
                messages=client.messages,
                model="claude-sonnet-4-5-20250929",
                provider="anthropic",
                system_prompt=client.system_prompt,
                temperature=client.temperature,
                max_tokens=client.max_tokens
            )

            await callback.message.edit_text(
                "✅ Переключено на 🧠 Claude Sonnet 4.5\n\nИстория сохранена.",
                reply_markup=get_main_keyboard()
            )

        elif callback_data == "clear_context":
            # Очистка контекста и статистики для текущей модели
            current_provider = context_manager.get_user_provider(user_id)
            context_manager.clear_context(user_id)

            # Сбрасываем статистику для текущей модели
            context_manager.reset_tokens_used(user_id, current_provider)

            logger.info(f"Пользователь {user_id} очистил контекст и статистику {current_provider}")

            model_name = "GPT" if current_provider == "openai" else "Claude"
            await callback.message.edit_text(
                f"🧹 Контекст очищен!\n📊 Статистика {model_name} сброшена!\n\nНачнем разговор заново.",
                reply_markup=get_main_keyboard()
            )

        elif callback_data == "show_info":
            # Показать информацию о текущей сессии
            current_model = context_manager.get_user_model(user_id)
            current_provider = context_manager.get_user_provider(user_id)
            current_temp = context_manager.get_user_temperature(user_id)
            current_tokens = context_manager.get_user_max_tokens(user_id)
            current_system = context_manager.get_user_system_prompt(user_id)
            messages_count = len(context_manager.get_user_messages(user_id))

            # Форматируем отображение system message
            if current_system:
                system_display = current_system[:50]
                if len(current_system) > 50:
                    system_display += "..."
                system_status = f"\"{system_display}\""
            else:
                system_status = "По умолчанию"

            model_name = "GPT-3.5-turbo" if current_provider == "openai" else "Claude Sonnet"
            provider_name = "OpenAI" if current_provider == "openai" else "Anthropic"

            info_text = (
                f"ℹ️ Информация о сессии:\n\n"
                f"🎯 Модель: {model_name}\n"
                f"🏢 Провайдер: {provider_name}\n"
                f"🌡️ Температура: {current_temp}\n"
                f"📏 Max tokens: {current_tokens}\n"
                f"💬 System message: {system_status}\n"
                f"💬 Сообщений в контексте: {messages_count}\n"
                f"🔑 Модель API: {current_model}"
            )

            await callback.message.edit_text(
                info_text,
                reply_markup=get_back_keyboard()
            )

        elif callback_data == "show_stats":
            # Показать статистику использования токенов
            logger.info(f"🔍 STATS: Запрос статистики от пользователя {user_id}")
            
            openai_tokens = context_manager.get_tokens_used(user_id, "openai")
            logger.info(f"🔍 STATS: OpenAI tokens = {openai_tokens} (type: {type(openai_tokens)})")
            
            anthropic_tokens = context_manager.get_tokens_used(user_id, "anthropic")
            logger.info(f"🔍 STATS: Anthropic tokens = {anthropic_tokens} (type: {type(anthropic_tokens)})")
            
            total_tokens = openai_tokens + anthropic_tokens
            logger.info(f"🔍 STATS: Total tokens = {total_tokens}")

            current_provider = context_manager.get_user_provider(user_id)
            current_model_name = "GPT-3.5-turbo" if current_provider == "openai" else "Claude Sonnet"

            logger.info(f"Показ статистики для пользователя {user_id}: GPT={openai_tokens}, Claude={anthropic_tokens}")

            stats_text = (
                f"📊 Статистика использования токенов:\n\n"
                f"🤖 GPT-3.5-turbo: {openai_tokens:,} токенов\n"
                f"🧠 Claude Sonnet: {anthropic_tokens:,} токенов\n"
                f"📈 Всего: {total_tokens:,} токенов\n\n"
                f"🎯 Текущая модель: {current_model_name}"
            )

            await callback.message.edit_text(
                stats_text,
                reply_markup=get_back_keyboard()
            )

        elif callback_data == "show_help":
            # Показать меню помощи
            help_text = (
                "❓ Помощь по использованию:\n\n"
                "🤖 Я - AI помощник с двумя моделями:\n"
                "• GPT-3.5-turbo - быстрый и универсальный\n"
                "• Claude Sonnet - думающий и аналитический\n\n"
                "💬 Просто пишите сообщения для общения!\n"
                "🔄 Переключайтесь между моделями\n"
                "🧹 Очищайте контекст при смене темы"
            )

            await callback.message.edit_text(
                help_text,
                reply_markup=get_help_keyboard()
            )

        elif callback_data == "show_commands":
            # Показать основные команды
            commands_text = (
                "🎯 Основные команды:\n\n"
                "📝 Просто пишите сообщения - бот ответит\n\n"
                "⌨️ Команды:\n"
                "/start - начать работу\n"
                "/switch_openai - выбрать GPT\n"
                "/switch_claude - выбрать Claude\n"
                "/clear - очистить контекст\n"
                "/help - эта справка\n\n"
                "🎮 Или используйте кнопки ниже:"
            )

            await callback.message.edit_text(
                commands_text,
                reply_markup=get_back_keyboard()
            )

        elif callback_data == "show_models_info":
            # Показать информацию о моделях
            models_text = (
                "📚 Информация о моделях:\n\n"
                "🤖 GPT-3.5-turbo (OpenAI):\n"
                "• Быстрый отклик\n"
                "• Универсальные задачи\n"
                "• Хорошо для диалога\n\n"
                "🧠 Claude Sonnet (Anthropic):\n"
                "• Глубокий анализ\n"
                "• Показывает размышления\n"
                "• Лучше для сложных задач\n\n"
                "💡 Совет: Для творчества - GPT,\n"
                "для анализа - Claude"
            )

            await callback.message.edit_text(
                models_text,
                reply_markup=get_back_keyboard()
            )

        elif callback_data == "show_settings":
            # Показать меню настроек
            current_temp = context_manager.get_user_temperature(user_id)
            current_tokens = context_manager.get_user_max_tokens(user_id)
            current_system = context_manager.get_user_system_prompt(user_id)

            # Форматируем отображение system message
            if current_system:
                system_display = current_system[:30]
                if len(current_system) > 30:
                    system_display += "..."
                system_status = f"\"{system_display}\""
            else:
                system_status = "По умолчанию"

            settings_text = (
                "⚙️ Настройки AI\n\n"
                f"🌡️ Температура: {current_temp}\n"
                f"📏 Max tokens: {current_tokens}\n"
                f"💬 System message: {system_status}\n\n"
                "Выберите параметр для изменения:"
            )

            await callback.message.edit_text(
                settings_text,
                reply_markup=get_settings_keyboard()
            )

        elif callback_data == "set_temperature":
            # Показать выбор температуры
            await callback.message.edit_text(
                "🌡️ Выберите температуру генерации:\n\n"
                "• 🎯 0.0 - максимально точный и предсказуемый ответ\n"
                "• ⚖️ 0.7 - сбалансированный режим (рекомендуется)\n"
                "• 🎨 1.0 - творческий режим\n"
                "• 🔥 1.5 - экспериментальный режим",
                reply_markup=get_temperature_keyboard()
            )

        elif callback_data.startswith("temp_"):
            # Установка температуры
            temp_value = float(callback_data.split("_")[1])
            client = get_ai_client(user_id)

            # Обновляем контекст с новой температурой
            context_manager.update_context(
                user_id=user_id,
                messages=client.messages,
                model=client.model,
                provider=client.provider,
                system_prompt=client.system_prompt,
                temperature=temp_value,
                max_tokens=client.max_tokens
            )

            await callback.message.edit_text(
                f"✅ Температура установлена на {temp_value}\n\n"
                "Настройки сохранены.",
                reply_markup=get_settings_keyboard()
            )

        elif callback_data == "set_max_tokens":
            # Показать выбор max_tokens
            await callback.message.edit_text(
                "📏 Выберите максимальное количество токенов:\n\n"
                "• 💬 500 - короткие ответы\n"
                "• 📝 1000 - стандартная длина\n"
                "• 📚 2000 - подробные ответы\n"
                "• 🎯 4000 - максимальная длина",
                reply_markup=get_max_tokens_keyboard()
            )

        elif callback_data.startswith("tokens_"):
            # Установка max_tokens
            tokens_value = int(callback_data.split("_")[1])
            client = get_ai_client(user_id)

            # Обновляем контекст с новыми max_tokens
            context_manager.update_context(
                user_id=user_id,
                messages=client.messages,
                model=client.model,
                provider=client.provider,
                system_prompt=client.system_prompt,
                temperature=client.temperature,
                max_tokens=tokens_value
            )

            await callback.message.edit_text(
                f"✅ Max tokens установлено на {tokens_value}\n\n"
                "Настройки сохранены.",
                reply_markup=get_settings_keyboard()
            )

        elif callback_data == "set_system_message":
            # Установка системного сообщения
            set_user_state(user_id, "waiting_system_message")
            await callback.message.edit_text(
                "💬 **Настройка системного сообщения**\n\n"
                "Отправьте текст, который будет определять поведение AI.\n\n"
                "📝 **Примеры:**\n"
                "• \"Ты - полезный помощник по Python\"\n"
                "• \"Отвечай кратко и по делу\"\n"
                "• \"Ты - эксперт по машинному обучению\"\n\n"
                "❌ Отправьте **-** для сброса к значению по умолчанию.\n\n"
                "После отправки вы вернетесь в меню настроек.",
                reply_markup=get_back_keyboard()
            )

        elif callback_data == "reset_settings":
            # Сброс настроек к значениям по умолчанию
            client = get_ai_client(user_id)

            context_manager.update_context(
                user_id=user_id,
                messages=client.messages,
                model=client.model,
                provider=client.provider,
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                temperature=DEFAULT_TEMPERATURE,
                max_tokens=DEFAULT_MAX_TOKENS
            )

            await callback.message.edit_text(
                f"✅ Настройки сброшены к значениям по умолчанию:\n"
                f"🌡️ Температура: {DEFAULT_TEMPERATURE}\n"
                f"📏 Max tokens: {DEFAULT_MAX_TOKENS}\n"
                f"💬 System message: \"{DEFAULT_SYSTEM_PROMPT}\"",
                reply_markup=get_settings_keyboard()
            )

        elif callback_data == "back_to_settings":
            # Возврат к меню настроек
            current_temp = context_manager.get_user_temperature(user_id)
            current_tokens = context_manager.get_user_max_tokens(user_id)
            current_system = context_manager.get_user_system_prompt(user_id)

            settings_text = (
                "⚙️ Настройки AI\n\n"
                f"🌡️ Температура: {current_temp}\n"
                f"📏 Max tokens: {current_tokens}\n"
                f"💬 System message: {'Установлено' if current_system else 'По умолчанию'}\n\n"
                "Выберите параметр для изменения:"
            )

            await callback.message.edit_text(
                settings_text,
                reply_markup=get_settings_keyboard()
            )

        elif callback_data == "back_to_main":
            # Возврат к главному меню
            current_model = context_manager.get_user_model(user_id)
            current_provider = context_manager.get_user_provider(user_id)
            model_name = "GPT-3.5-turbo" if current_provider == "openai" else "Claude Sonnet"

            main_text = (
                "🏠 Главное меню\n\n"
                f"🎯 Текущая модель: {model_name}\n\n"
                "Выберите действие:"
            )

            await callback.message.edit_text(
                main_text,
                reply_markup=get_main_keyboard()
            )

        # Подтверждаем обработку callback
        await callback.answer()

    except Exception as e:
        logger.error(f"Ошибка при обработке callback {callback_data} от пользователя {user_id}: {str(e)}")
        await callback.answer("❌ Произошла ошибка", show_alert=True)


async def main():
    """
    Основная функция запуска бота.
    """
    logger.info("Запуск AI Telegram бота...")

    try:
        # Запускаем polling
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {str(e)}")
        raise
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
