# Analitik.Lab AI Bot

AI-ассистент для лаборатории **«Аналитик.Лаб»** — Telegram-бот, который принимает заявки клиентов, задаёт уточняющие вопросы и автоматически формирует коммерческие предложения.

## Что делает бот

```
Клиент пишет в Telegram → Supervisor определяет intent
  → Intake-агент уточняет детали (тип анализа, адрес, сроки)
  → Proposal-агент формирует КП в DOCX
  → Бот отправляет готовый файл клиенту
```

Среднее время обработки запроса: **3–7 минут** вместо 30–60 минут менеджером.

## Архитектура

```
Telegram (aiogram 3.x)
    │
    ▼
LangGraph StateGraph
    │
    ├── Supervisor Agent (GigaChat 2 Pro) — маршрутизация
    ├── Intake Agent (GigaChat 2 Lite) — уточняющие вопросы
    └── Proposal Agent (GigaChat 2 Lite) — генерация КП
            │
            ▼
        PDF Generator (docxtpl)
```

- **Только российские LLM** — GigaChat 2 (Sber), соответствие 152-ФЗ
- **Stateful-диалог** — LangGraph MemorySaver хранит контекст каждого клиента
- **Генерация КП** — DOCX из шаблона через Jinja2 (docxtpl)

## Структура проекта

```
src/
├── main.py              # Точка входа: запуск бота
├── config.py            # Конфигурация (pydantic-settings)
├── bot/
│   ├── handlers.py      # Хэндлеры /start и сообщений
│   └── middleware.py     # Логирование, rate-limit, обработка ошибок
├── agents/
│   ├── supervisor.py    # Supervisor Agent — анализ intent, маршрутизация
│   ├── intake.py        # Intake Agent — сбор данных от клиента
│   └── proposal.py      # Proposal Agent — формирование КП
├── graphs/
│   └── main_graph.py    # LangGraph StateGraph (supervisor → intake → proposal)
├── tools/
│   └── pdf_generator.py # Генерация DOCX из шаблона
├── knowledge/
│   ├── proposal_template.docx       # Jinja2-шаблон КП (используется сейчас)
│   ├── examples/
│   │   ├── kp/                      # 8 реальных PDF КП (few-shot примеры)
│   │   └── requests/                # Заявки на отбор проб (DOCX + PDF)
│   ├── samples/
│   │   └── application_template.docx  # Образец заявки
│   └── prices/                      # Прейскурант (будет добавлен)
└── schemas/
    └── state.py         # AgentState, IntakeData, ProposalData
```

## Knowledge Base

Папка `src/knowledge/` содержит базу знаний для AI-агентов.

| Папка | Содержимое | Использование |
|-------|-----------|---------------|
| `proposal_template.docx` | Jinja2-шаблон КП | Фаза 1 — генерация DOCX через docxtpl |
| `examples/kp/` | 8 реальных PDF коммерческих предложений | Фаза 1 — few-shot примеры в prompt Proposal Agent |
| `examples/requests/` | Заявки на отбор проб (DOCX + PDF) | Фаза 2 — обучение Intake Agent |
| `samples/` | Образец заявки | Фаза 2 — шаблон для генерации заявок |
| `prices/` | Прейскурант услуг | Фаза 2 — RAG (Qdrant) для расчёта стоимости |

**Фаза 1 (сейчас):** Proposal Agent формирует JSON-структуру КП на основе `intake_data` и `few-shot` примеров из `examples/kp/`, затем рендерит документ через `proposal_template.docx`.

Few-shot механизм:
- Загружаются PDF из `src/knowledge/examples/kp/`
- Выбираются до 3 самых релевантных примеров по лексическому совпадению с запросом клиента
- При ошибке чтения PDF агент продолжает работу без few-shot (graceful fallback)

**Фаза 2 (планируется):** `prices/` будет индексирован в Qdrant для точного расчёта стоимости через RAG.

## Быстрый старт

### Требования

- Python 3.11+
- Telegram Bot Token ([BotFather](https://t.me/BotFather))
- GigaChat API credentials ([developers.sber.ru](https://developers.sber.ru/))

### Установка

```bash
git clone https://github.com/VasiliiLbyte/analitik_lab.git
cd analitik_lab
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -e ".[dev]"
```

### Настройка

```bash
cp .env.example .env
```

Заполните `.env`:

```ini
TELEGRAM_BOT_TOKEN=ваш-токен-от-botfather
GIGACHAT_CREDENTIALS=ваши-credentials-от-sber
GIGACHAT_SCOPE=GIGACHAT_API_PERS
LOG_LEVEL=INFO
```

### Запуск

```bash
python -m src.main
```

## Тестирование

```bash
# Все тесты
python -m pytest

# С покрытием
python -m pytest --cov=src --cov-report=term-missing

# Конкретный модуль
python -m pytest tests/test_agents/ -v
```

Текущее покрытие: **85%** (77 тестов).

## Технологии

| Компонент | Технология | Версия |
|-----------|------------|--------|
| Бот | aiogram | 3.26+ |
| Оркестрация агентов | LangGraph | 1.1+ |
| LLM | GigaChat 2 (Lite / Pro) | через langchain-gigachat 0.5+ |
| Генерация КП | docxtpl | 0.18+ |
| Валидация | pydantic | 2.x |
| Логирование | loguru | 0.7+ |
| Тестирование | pytest + pytest-asyncio | 8.x |

## Фазы разработки

### Фаза 1 — MVP (текущая)

- [x] Telegram-бот (aiogram 3.x)
- [x] Supervisor Agent — маршрутизация intent
- [x] Intake Agent — уточняющие вопросы
- [x] Proposal Agent — генерация КП
- [x] LangGraph StateGraph с памятью диалога
- [x] Генерация DOCX из шаблона
- [x] TDD, 85% покрытие

### Фаза 2 — CRM + Объяснения (планируется)

- [ ] CRM Agent — интеграция с Bitrix24 (Lead → Deal → Task)
- [ ] Explanation Agent — объяснение результатов анализов простым языком
- [ ] Qdrant RAG — база знаний (методики, прайсы, нормативы)
- [ ] Max-бот (VK Teams)
- [ ] Email-канал

## Безопасность

- Секреты только через переменные окружения (`.env`)
- Валидация всех входящих сообщений (длина, формат)
- Rate-limiting: 10 сообщений/мин на пользователя
- Логирование без PII (только `chat_id` и длина сообщения)
- Только российские LLM-провайдеры (152-ФЗ)

## Лицензия

Проприетарный проект ООО «Аналитик.Лаб». Все права защищены.
