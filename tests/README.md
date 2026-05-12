# Tests for Earth-Agent Multi-Agent System

Этот каталог содержит unit и integration тесты для мультиагентной системы.

## 🎯 Цель

Тестировать код **БЕЗ трат токенов OpenAI** используя:
- **FakeListChatModel** (встроенный в LangChain) - для мокирования LLM
- **unittest.mock** - для мокирования MCP tools
- **pytest** - как test runner

## 📦 Установка зависимостей

```bash
# Основные зависимости уже должны быть установлены из requirements.txt
# Дополнительно для тестирования:
pip install pytest pytest-asyncio
```

## 🚀 Запуск тестов

### Все тесты:
```bash
pytest tests/ -v
```

### Конкретный тест:
```bash
pytest tests/test_multiagent_system.py::test_location_agent_finds_coordinates -v
```

### С выводом print:
```bash
pytest tests/ -v -s
```

### С coverage:
```bash
pytest tests/ --cov=scripts --cov-report=html
```

## 📋 Структура тестов

### Unit Tests (изолированные компоненты) - `test_multiagent_system.py`:

**Location Agent:**
- ✅ `test_location_agent_finds_coordinates` - поиск координат
- ✅ `test_location_agent_parse_coordinates` - парсинг результатов

**Data Acquisition Agent:**
- ✅ `test_data_acquisition_agent_downloads_files` - скачивание файлов
- ✅ `test_data_acquisition_agent_parse_downloaded_files` - парсинг путей

**Supervisor Routing:**
- ✅ `test_supervisor_routing_with_location` - routing к Location Agent
- ✅ `test_supervisor_routing_with_data_acquisition` - routing к Data Acquisition Agent
- ✅ `test_supervisor_routing_direct_to_main` - прямой routing к Main Agent

### Integration Tests - `test_multiagent_system.py`:

- ✅ `test_supervisor_full_workflow` - полный цикл от вопроса до ответа

### E2E Tests (end-to-end с реальным кодом) - `test_e2e_multiagent.py`:

**Full Workflow:**
- ✅ `test_e2e_supervisor_coordinates_data_analysis` - полный E2E тест с реальным Supervisor кодом

**Logging Tests:**
- ✅ `test_e2e_location_agent_logging` - проверка логирования Location Agent
- ✅ `test_e2e_data_acquisition_logging_retry` - проверка логирования retry попыток
- ✅ `test_e2e_log_files_created` - проверка создания файлов логов с UTF-8

**Error Handling:**
- ✅ `test_e2e_supervisor_handles_location_failure` - обработка ошибок Location Agent

**Routing (Parametrized):**
- ✅ `test_e2e_supervisor_routing_logic` - 4 параметризованных теста routing логики

## 🔧 Как это работает

### 1. FakeListChatModel для LLM

```python
from langchain_core.language_models.fake_chat_models import FakeListChatModel

# Создаём фейковый LLM с заранее заготовленными ответами
mock_llm = FakeListChatModel(responses=[
    '{"location_needed": true, ...}',
    'Найдены координаты: ...'
])
```

### 2. StructuredTool для мокирования MCP tools

```python
from langchain.tools import StructuredTool

def mock_search_location(**kwargs):
    return '[{"lat": "55.5", "lon": "37.5", ...}]'

mock_tool = StructuredTool.from_function(
    func=mock_search_location,
    name="search_location",
    description="Mock OSM search"
)
```

### 3. Передача моков в агентов

```python
# Агенты принимают те же аргументы, просто с фейковыми объектами
location_agent = LocationAgent(
    llm=mock_llm,           # FakeListChatModel
    osm_tools=mock_tools,   # Mock StructuredTools
    max_iterations=5
)
```

**Никаких изменений в production коде!**

## ✅ Преимущества подхода

1. **Нет трат токенов** - все LLM ответы заранее записаны
2. **Быстрые тесты** - нет реальных API запросов
3. **Воспроизводимость** - одинаковые результаты каждый раз
4. **Минимальные изменения кода** - используем те же классы, просто с моками
5. **Официальная поддержка** - FakeListChatModel это часть LangChain Core

## 🐛 Отладка тестов

Если тест падает:

1. Запустите с `-v -s` для вывода логов
2. Проверьте что responses в FakeListChatModel соответствуют ожидаемым вызовам
3. Используйте `print()` для отладки (они выведутся с `-s`)

## 📚 Дополнительные ресурсы

- [LangChain Testing Guide](https://python.langchain.com/docs/contributing/testing/)
- [pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Guide](https://docs.python.org/3/library/unittest.mock.html)
