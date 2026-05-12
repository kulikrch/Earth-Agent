# Как Работают Скрипты `evaluate`

## Назначение
Папка `evaluate` нужна для проверки качества запусков агента на датасете `benchmark/development_quest.json`.

Проверка делится на два уровня:
1. `step_by_step` — насколько траектория вызовов инструментов похожа на эталон.
2. `end_to_end` — правильность финального ответа и эффективность по числу шагов.

## Откуда берутся данные
Для одного запуска (например, `evaluate_langchain/gpt-4o_dev_IF_26-05-07_00-10`) используются такие файлы:

1. `gpt-4o_IF_langchain.json`
- Сырые логи диалога/инструментов.
- Источник фактических tool-calls.

2. `results_summary.json`
- Сырые финальные ответы модели по вопросам.

3. `benchmark/development_quest.json`
- Ground truth:
  - правильные ответы (`gt_answer.whitelist`);
  - ожидаемые tool-цепочки (`dialogs` и/или `expected_tool_chains`).

## Какие промежуточные файлы создаются
Скрипты автоматически создают недостающие нормализованные артефакты:

1. `extracted_tool_calls.json`
- Строится из `gpt-4o_IF_langchain.json`.
- Содержит унифицированную структуру вызовов tools.
- Нужен, потому что исходные conversation-логи бывают неоднородными.

2. `results_summary_polished.json`
- Строится из `results_summary.json`.
- Приводит ответы к единому виду (`<Answer>A</Answer>` и т.п.).

## Логика `step_by_step.py`
Сравнивает фактические вызовы инструментов с эталоном и считает 4 метрики:

1. `contains_all_tool_calls_any_order`
- Проверяет, что нужные инструменты вообще встретились (без учёта порядка).

2. `contains_all_tool_calls_in_order`
- Проверяет относительный порядок вызовов.

3. `trajectory_step_wise_score`
- Проверяет длину правильного префикса шагов (первые шаги подряд).

4. `parameter_accuracy`
- Сравнивает имя инструмента + параметры пошагово.
- Останавливается на первом рассогласовании (чтобы ранняя ошибка была явно видна).

Для multi-agent сценария:
- сравнение делается отдельно для `location_agent`, `data_acquisition_agent`, `main_agent`;
- если для агента задано несколько валидных стратегий (`any_of`), берётся лучшая.

Результат сохраняется в:
- `step_by_step_evaluation_results.json`

## Математика `step_by_step`
Обозначения для одного вопроса:
- `E = [e1, e2, ..., en]` — ожидаемая последовательность имён tools.
- `A = [a1, a2, ..., am]` — фактическая последовательность имён tools.
- Для multi-agent: `E_g`, `A_g` — те же последовательности для агента `g`.

### 1) `contains_all_tool_calls_any_order`
В коде сравнение идёт по множествам (дубликаты игнорируются):
- `S_E = set(E)`, `S_A = set(A)`
- `score = |S_E ∩ S_A| / |S_E|`

### 2) `contains_all_tool_calls_in_order`
Считается длина максимального совпавшего подпоследовательного порядка:
- `matched` — сколько элементов из `E` найдено в `A` в правильном относительном порядке.
- `score = matched / |E|`

### 3) `trajectory_step_wise_score`
Это префиксная метрика:
- `k` — длина общего префикса `E` и `A` до первого несовпадения.
- `score = k / |E|`

### 4) `parameter_accuracy`
Пошаговое сравнение `(tool_name, input)`:
- шаг считается верным, если:
  - `name_match = (expected_name == actual_name)`, и
  - `input_match = True`, если у эталона `input` пустой (`{}`/`None`/`[]`),
    иначе `input_match = (expected_input == actual_input)` (строгое равенство).
- Счёт останавливается на первом неверном шаге.
- `score = matched_steps / |expected_calls|`

### Агрегация в multi-agent режиме
Пусть у агента `g` есть набор допустимых цепочек `C_g = {c1, c2, ...}` (`any_of`).
Для каждой метрики:
1. Считаем score для всех `c ∈ C_g`.
2. Берём лучший: `score_g = max_c metric(A_g, c)`.
3. Усредняем только по агентам с `required=true`:
   - `final_score = (1/R) * Σ score_g`, где `R` — число required-агентов.

Если у required-агентов нет валидных candidate-цепочек, в коде используется защитное значение `1.0`.

## Логика `end_to_end.py`
Считает две группы метрик:

1. Accuracy
- Сравнивает букву ответа модели с эталонной буквой из benchmark.

2. Efficiency
- Формула: `model_tool_count / expected_tool_count`.
- `expected_tool_count` берётся из эталона:
  - для multi-agent: сумма минимальных длин required-цепочек;
  - иначе: длина `expected_tool_calls`.

Результат сохраняется в:
- `end_to_end_evaluation_results.json`

## Математика `end_to_end`
Для набора из `N` вопросов:

### 1) Accuracy
- `evaluated_questions` — сколько вопросов реально имеют предсказание.
- `correct_answers` — сколько совпало с эталонной буквой.
- `accuracy_rate = correct_answers / evaluated_questions`

Дополнительно:
- `fail_rate = fail_answers / evaluated_questions`

### 2) Efficiency
Для вопроса `i`:
- `model_tool_count_i` — число фактических tool-calls.
- `expected_tool_count_i` — эталонная длина.
- `efficiency_i = model_tool_count_i / expected_tool_count_i`

Особый случай в коде:
- если `expected_tool_count_i = 0`:
  - `efficiency_i = 1.0`, если `model_tool_count_i = 0`;
  - `efficiency_i = inf`, если `model_tool_count_i > 0`.

Итог:
- `average_efficiency = mean(efficiency_i)` по вопросам, где есть фактические tool-calls.

### Как считается `expected_tool_count`
- Multi-agent: сумма минимумов по required-агентам:
  - `expected_tool_count = Σ_g min(len(c) for c in any_of_g)`
- Single/legacy: `len(expected_tool_calls)`.

## Мини-пример сравнения
Пусть эталон:
- `E = [get_filelist, calculate_batch_ndbi, calculate_threshold_ratio]`

Факт:
- `A = [calculate_batch_ndbi, calculate_threshold_ratio, percentage_change]`

Тогда:
- `contains_all_tool_calls_any_order = 2/3` (совпали 2 имени из 3)
- `contains_all_tool_calls_in_order = 2/3` (в порядке найдены `calculate_batch_ndbi`, `calculate_threshold_ratio`)
- `trajectory_step_wise_score = 0/3` (первый же шаг не совпал с `get_filelist`)
- `parameter_accuracy` также начнёт с шага 1 и остановится на первом несовпадении.

## Быстрые команды
Один запуск:

```powershell
.\.venv\Scripts\python.exe evaluate/step_by_step.py --run-dir evaluate_langchain/<RUN_DIR> --benchmark benchmark/development_quest.json
.\.venv\Scripts\python.exe evaluate/end_to_end.py --run-dir evaluate_langchain/<RUN_DIR> --benchmark benchmark/development_quest.json
```

Batch по всем запускам:

```powershell
.\.venv\Scripts\python.exe evaluate/step_by_step.py
.\.venv\Scripts\python.exe evaluate/end_to_end.py
```

## Почему это важно
Разделение на `step_by_step` и `end_to_end` помогает видеть оба типа проблем:
1. Агент дал правильный финальный ответ, но пришёл к нему не тем путём (инструменты/параметры).
2. Агент шёл по корректной траектории, но ошибся в интерпретации и финальном выборе.
