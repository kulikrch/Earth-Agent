#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Утилиты для валидации бенчмарка.

Этот модуль связывает три источника:
1) `benchmark/development_quest.json` — эталон (правильный ответ + ожидаемые цепочки tools),
2) артефакты запуска в `evaluate_langchain/<run_dir>/` — фактическое поведение агента,
3) вычисление метрик для `step_by_step.py` и `end_to_end.py`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def normalize_question_index(value: Any) -> str:
    """Нормализует идентификатор вопроса к формату `questionN`.

    Зачем: в разных файлах один и тот же вопрос может называться как `2` или `question2`.
    Для корректного join между эталоном и предиктами нужен единый ключ.
    """
    s = str(value).strip()
    if s.startswith("question"):
        return s
    if s.isdigit():
        return f"question{s}"
    return s


def denormalize_question_id(question_index: str) -> str:
    """Обратное преобразование `questionN` -> `N` (где это требуется во внешних файлах)."""
    if question_index.startswith("question"):
        return question_index[len("question") :]
    return question_index


def load_json(path: Path) -> Any:
    """Читает JSON из файла в UTF-8.

    Используется для всех типов входов:
    - benchmark,
    - conversation logs,
    - results_summary,
    - промежуточные extracted файлы.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    """Сохраняет JSON (UTF-8, pretty-print) и при необходимости создаёт директорию."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_answer_letter(text: Any) -> str:
    """Извлекает букву ответа (A/B/C/...) из сырых ответов модели.

    Почему это нужно:
    - модель может вернуть `<Answer>A</Answer>`, `A.`, `Ваш выбор (A)` и т.д.;
    - иногда встречаются кириллические буквы (`А/В/С/Д`), их нужно привести к латинице;
    - при ошибке или нераспознанном формате возвращаем `FAIL`/`UNKNOWN`, чтобы
      метрика точности была прозрачной.
    """
    if text is None:
        return "FAIL"
    raw = str(text)
    if "FAIL" in raw.upper():
        return "FAIL"

    # Нормализуем распространённые кириллические буквы вариантов к латинице.
    letter_map = {
        "А": "A",
        "В": "B",
        "С": "C",
        "Д": "D",
        "Е": "E",
        "Ф": "F",
    }
    for cyr, lat in letter_map.items():
        raw = raw.replace(cyr, lat)

    # Сначала разбираем содержимое тега <Answer>. Если внутри не одна буква,
    # а текст вроде "В 2023 ... (C)", нельзя брать первую кириллическую "В"
    # как вариант B. Приоритет у явных маркеров варианта внутри тега.
    answer_tag = re.search(r"<Answer>\s*(.*?)\s*</Answer>", raw, re.IGNORECASE | re.DOTALL)
    if answer_tag:
        inner = answer_tag.group(1).strip()
        for pattern in (
            r"^\s*([A-F])\s*$",
            r"\(\s*([A-F])\s*\)",
            r"(?:вариант|option|answer|ответ|выбор)\s*[:\-]?\s*([A-F])\b",
            r"^\s*([A-F])\s*[\.\):\-]",
        ):
            m = re.search(pattern, inner, re.IGNORECASE)
            if m:
                return m.group(1).upper()

    # Частые прямые форматы: "A....", "A)", "Ваш выбор (A)".
    direct_patterns = (
        r"\(\s*([A-F])\s*\)",
        r"^\s*([A-F])\s*[\.\):\-]",
        r"ваш\s+выбор\s*\(\s*([A-F])\s*\)",
    )
    for pattern in direct_patterns:
        m = re.search(pattern, raw, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    matches = re.findall(r"\b([A-F])\b", raw, flags=re.IGNORECASE)
    if matches:
        return matches[0].upper()

    return "UNKNOWN"


def _tool_output_to_text(content: Any) -> str:
    """Приводит output tool-сообщения к строке.

    В логах output бывает строкой, списком структур OpenAI/LangChain или словарём.
    Для единообразного сравнения и сохранения в extracted-слой всё сводим к тексту.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        texts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                output = item.get("output")
                if isinstance(output, list):
                    for output_item in output:
                        if isinstance(output_item, dict) and "text" in output_item:
                            texts.append(str(output_item["text"]))
                elif "text" in item:
                    texts.append(str(item["text"]))
            else:
                texts.append(str(item))
        return "\n".join(texts).strip()
    return str(content)


def _extract_assistant_tool_calls(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Извлекает tool-calls из assistant-сообщения.

    Поддерживаются оба формата, встречающиеся в логах:
    - `content = [{"name": ..., "input": ...}]`
    - `tool_calls = [{"function": {"name": ..., "arguments": ...}}]`
    """
    calls: List[Dict[str, Any]] = []

    # Формат, встречающийся в логах запуска:
    # assistant.content = [{"name": "...", "input": {...}}, ...]
    content = message.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and "name" in item and "input" in item:
                calls.append({"name": item["name"], "input": item.get("input", {})})

    # Альтернативный формат: assistant.tool_calls (OpenAI-подобные вызовы функций).
    tool_calls = message.get("tool_calls", [])
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            args = function.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            if name:
                calls.append({"name": name, "input": args})

    return calls


def extract_tool_calls_from_langchain_log(log_path: Path) -> List[Dict[str, Any]]:
    """Строит нормализованный список tool-calls из `*_langchain.json`.

    Откуда данные:
    - вход: `evaluate_langchain/<run_dir>/gpt-4o_IF_langchain.json`
    - выход: `extracted_tool_calls.json` (используется оценщиками).

    Почему нужен отдельный extracted-слой:
    - исходные conversation-логи неоднородны по структуре;
    - метрикам удобнее работать с унифицированной схемой `{name,input,output}`.
    """
    raw = load_json(log_path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array in {log_path}, got {type(raw)}")

    results: List[Dict[str, Any]] = []
    for record in raw:
        if not isinstance(record, dict):
            continue

        qidx = normalize_question_index(record.get("question_index", ""))
        conversations = record.get("conversations", [])
        tool_calls: List[Dict[str, Any]] = []
        tool_calls_by_agent: Dict[str, List[Dict[str, Any]]] = {
            "location_agent": [],
            "data_acquisition_agent": [],
            "main_agent": [],
        }

        if isinstance(conversations, list):
            for message in conversations:
                if not isinstance(message, dict):
                    continue
                role = message.get("role")

                if role == "assistant":
                    for tool_call in _extract_assistant_tool_calls(message):
                        tool_calls.append(
                            {
                                "name": tool_call.get("name", ""),
                                "input": tool_call.get("input", {}),
                                "output": None,
                            }
                        )
                elif role == "tool":
                    tool_name = message.get("name", "")
                    tool_output = _tool_output_to_text(message.get("content"))
                    # Привязываем результат к последнему "незакрытому" вызову такого же инструмента.
                    # Это важно, когда один и тот же tool вызывается несколько раз подряд.
                    matched = False
                    for pending in reversed(tool_calls):
                        if pending["name"] == tool_name and pending["output"] is None:
                            pending["output"] = tool_output
                            matched = True
                            break
                    if not matched:
                        tool_calls.append({"name": tool_name, "input": {}, "output": tool_output})

                    # Представление для мультиагентного режима: сохраняем разрез по агентам
                    # (location/data_acq/main),
                    # чтобы далее сравнивать траектории по каждому агенту отдельно.
                    agent_key = _normalize_agent_key(message.get("agent"))
                    if agent_key not in tool_calls_by_agent:
                        tool_calls_by_agent[agent_key] = []
                    if tool_name:
                        tool_calls_by_agent[agent_key].append(
                            {"name": tool_name, "input": {}, "output": tool_output}
                        )

        completed = [tc for tc in tool_calls if tc["name"]]
        results.append(
            {
                "question_index": qidx,
                "query": "",
                "tool_calls": completed,
                "tool_calls_by_agent": tool_calls_by_agent,
                "final_answer": record.get("final_answer", ""),
            }
        )

    return results


def polish_results_summary(results_summary_path: Path) -> List[Dict[str, Any]]:
    """Нормализует итоговые ответы модели в единый формат.

    Откуда данные:
    - вход: `results_summary.json` (сырые ответы после прогона),
    - выход: `results_summary_polished.json`.

    Зачем:
    - end-to-end accuracy должна сравнивать букву ответа, а не произвольный текст.
    """
    rows = load_json(results_summary_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {results_summary_path}")

    polished: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("question_id", "")).strip()
        answer_text = row.get("answer", "")
        letter = extract_answer_letter(answer_text)
        if letter in {"A", "B", "C", "D", "E", "F"}:
            final_answer = f"<Answer>{letter}</Answer>"
        elif letter == "FAIL":
            final_answer = "FAIL"
        else:
            final_answer = str(answer_text)

        polished.append(
            {
                "question_id": qid,
                "original_answer": answer_text,
                "final_answer": final_answer,
            }
        )
    return polished


def _parse_expected_tool_calls(question_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Извлекает ожидаемые tool-calls для legacy/single-agent сценария.

    Приоритет:
    1) `dialogs[].assistant.tool_calls` (точная траектория),
    2) fallback на `tools` (если detailed dialogs отсутствуют).
    """
    expected_calls: List[Dict[str, Any]] = []

    dialogs = question_item.get("dialogs", [])
    if isinstance(dialogs, list):
        for msg in dialogs:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            for call in msg.get("tool_calls", []) or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function", {})
                if not isinstance(fn, dict):
                    continue
                name = fn.get("name")
                args = fn.get("arguments", {})
                if name:
                    expected_calls.append({"name": name, "input": args, "output": None})

    # Резервный вариант: если в dialogs нет tool-calls, используем список `tools`.
    if not expected_calls:
        tools = question_item.get("tools", [])
        if isinstance(tools, list):
            for tool_name in tools:
                expected_calls.append({"name": str(tool_name), "input": {}, "output": None})

    return expected_calls


def _normalize_agent_key(agent_key: Any) -> str:
    """Приводит разные варианты имени агента к каноническому ключу."""
    raw = str(agent_key or "").strip().lower()
    if raw in {"location", "location_agent"}:
        return "location_agent"
    if raw in {"data_acquisition", "data_acquisition_agent", "data-acquisition"}:
        return "data_acquisition_agent"
    if raw in {"main", "main_agent", ""}:
        return "main_agent"
    return raw


def _parse_candidate_calls(candidate: Any) -> List[Dict[str, Any]]:
    """Парсит один candidate-цепочку из `expected_tool_chains.multiagent.*.any_of`."""
    calls: List[Dict[str, Any]] = []
    if not isinstance(candidate, list):
        return calls
    for step in candidate:
        if isinstance(step, str):
            name = step.strip()
            if name:
                calls.append({"name": name, "input": {}, "output": None})
            continue
        if not isinstance(step, dict):
            continue
        name = str(step.get("name", "")).strip()
        if not name:
            continue
        calls.append({"name": name, "input": step.get("input", {}) or {}, "output": None})
    return calls


def _parse_expected_multiagent_chains(question_item: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Извлекает мультиагентные эталонные цепочки из benchmark.

    Возвращается структура:
    - agent -> {required: bool, any_of: [candidate1, candidate2, ...]}

    Почему `any_of`:
    - у агента может быть несколько корректных стратегий решения одного вопроса.
    """
    raw = question_item.get("expected_tool_chains", {})
    if not isinstance(raw, dict):
        return {}
    multi = raw.get("multiagent", {})
    if not isinstance(multi, dict):
        return {}

    parsed: Dict[str, Dict[str, Any]] = {}
    for agent_key, agent_spec in multi.items():
        if not isinstance(agent_spec, dict):
            continue
        normalized_key = _normalize_agent_key(agent_key)
        required = bool(agent_spec.get("required", True))
        any_of_raw = agent_spec.get("any_of", [])
        any_of: List[List[Dict[str, Any]]] = []
        if isinstance(any_of_raw, list):
            for candidate in any_of_raw:
                calls = _parse_candidate_calls(candidate)
                if calls:
                    any_of.append(calls)
        if any_of:
            parsed[normalized_key] = {"required": required, "any_of": any_of}
    return parsed


def load_dev_ground_truth(benchmark_path: Path) -> Dict[str, Dict[str, Any]]:
    """Собирает ground truth из `benchmark/development_quest.json`.

    Для каждого вопроса сохраняет:
    - нормализованный question_index,
    - expected_tool_calls (legacy),
    - expected_multiagent_chains (если заданы),
    - final_answer (буква правильного варианта).
    """
    benchmark = load_json(benchmark_path)
    if not isinstance(benchmark, dict):
        raise ValueError(f"Expected object in {benchmark_path}")

    gt: Dict[str, Dict[str, Any]] = {}
    for question_id, item in benchmark.items():
        if not isinstance(item, dict):
            continue
        qidx = normalize_question_index(question_id)
        eval_rows = item.get("evaluation", [])
        answer_letter = ""
        if isinstance(eval_rows, list) and eval_rows:
            first = eval_rows[0]
            if isinstance(first, dict):
                gt_answer = first.get("gt_answer", {})
                if isinstance(gt_answer, dict):
                    answer_letter = str(gt_answer.get("whitelist", "")).strip().upper()
        gt[qidx] = {
            "question_index": qidx,
            "expected_tool_calls": _parse_expected_tool_calls(item),
            "expected_multiagent_chains": _parse_expected_multiagent_chains(item),
            "final_answer": answer_letter,
        }
    return gt


def find_single_log_file(run_dir: Path) -> Path:
    """Ищет единственный `*_langchain.json` внутри run-dir."""
    candidates = sorted(run_dir.glob("*_langchain.json"))
    if not candidates:
        raise FileNotFoundError(f"No *_langchain.json file found in {run_dir}")
    return candidates[0]


def get_expected_tool_count(gt_item: Dict[str, Any]) -> int:
    """
    Базовое ожидаемое число шагов для efficiency-метрики.

    Логика:
    - для multi-agent берём минимальную длину среди `any_of` у каждого required-агента;
    - если multi-agent структура отсутствует, используем длину `expected_tool_calls`.
    Почему минимум:
    - это "оптимальная" эталонная стоимость решения без штрафа за альтернативные пути.
    """
    multi = gt_item.get("expected_multiagent_chains", {})
    if isinstance(multi, dict) and multi:
        total = 0
        for _, spec in multi.items():
            if not isinstance(spec, dict):
                continue
            if not bool(spec.get("required", True)):
                continue
            any_of = spec.get("any_of", [])
            if not isinstance(any_of, list) or not any_of:
                continue
            lengths = [len(candidate) for candidate in any_of if isinstance(candidate, list)]
            if lengths:
                total += min(lengths)
        if total > 0:
            return total
    return len(gt_item.get("expected_tool_calls", []))


def get_tool_name_sequence(tool_calls: List[Dict[str, Any]]) -> List[str]:
    """Возвращает только последовательность имён инструментов (без параметров/output)."""
    return [str(tc.get("name", "")) for tc in tool_calls if tc.get("name")]


def contains_all_tool_calls_any_order(actual_calls: List[Dict[str, Any]], expected_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Доля ожидаемых tools, которые вообще встретились (порядок не важен)."""
    expected = get_tool_name_sequence(expected_calls)
    actual = get_tool_name_sequence(actual_calls)
    if not expected:
        return {"score": 1.0, "details": {"matched_tools": 0, "total_expected": 0}}

    expected_set = set(expected)
    actual_set = set(actual)
    matched = expected_set.intersection(actual_set)
    return {
        "score": len(matched) / len(expected_set),
        "details": {
            "matched_tools": len(matched),
            "total_expected": len(expected_set),
            "matched_tool_names": sorted(matched),
            "expected_sequence": expected,
            "actual_sequence": actual,
        },
    }


def contains_all_tool_calls_in_order(actual_calls: List[Dict[str, Any]], expected_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Доля ожидаемых tools, найденных в правильном относительном порядке."""
    expected = get_tool_name_sequence(expected_calls)
    actual = get_tool_name_sequence(actual_calls)
    if not expected:
        return {"score": 1.0, "details": {"matched_in_order": 0, "total_expected": 0}}

    matched = 0
    actual_iter = iter(actual)
    for exp in expected:
        found = False
        for act in actual_iter:
            if act == exp:
                matched += 1
                found = True
                break
        if not found:
            break

    return {
        "score": matched / len(expected),
        "details": {
            "matched_in_order": matched,
            "total_expected": len(expected),
            "expected_sequence": expected,
            "actual_sequence": actual,
        },
    }


def trajectory_step_wise_score(actual_calls: List[Dict[str, Any]], expected_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Префиксная точность траектории: сколько первых шагов подряд совпало."""
    expected = get_tool_name_sequence(expected_calls)
    actual = get_tool_name_sequence(actual_calls)
    if not expected:
        return {"score": 1.0, "details": {"correct_steps": 0, "total_expected": 0}}

    correct = 0
    for exp, act in zip(expected, actual):
        if exp == act:
            correct += 1
        else:
            break
    return {
        "score": correct / len(expected),
        "details": {
            "correct_steps": correct,
            "total_expected": len(expected),
            "expected_sequence": expected,
            "actual_sequence_prefix": actual[: len(expected)],
        },
    }


def parameter_accuracy(actual_calls: List[Dict[str, Any]], expected_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Пошаговая точность (имя инструмента + параметры).

    Метрика останавливается на первом несовпадении:
    это делает ошибку в раннем критическом шаге более заметной.
    """
    if not expected_calls:
        return {
            "score": 1.0,
            "details": {
                "matched_steps": 0,
                "total_expected_steps": 0,
                "mode": "empty_expected",
            },
        }

    matched_steps = 0
    details: List[Dict[str, Any]] = []
    for idx, expected in enumerate(expected_calls):
        if idx >= len(actual_calls):
            details.append(
                {
                    "step": idx + 1,
                    "expected_tool_name": expected.get("name", ""),
                    "actual_tool_name": None,
                    "name_match": False,
                    "input_match": False,
                    "reason": "missing_step",
                }
            )
            continue

        actual = actual_calls[idx]
        exp_name = expected.get("name", "")
        act_name = actual.get("name", "")
        exp_input = expected.get("input", {})
        act_input = actual.get("input", {})

        name_match = exp_name == act_name
        # Если у эталона пустой input, сравниваем только имя инструмента.
        # Это делает метрику менее хрупкой для dev-набора.
        if exp_input in ({}, None, []):
            input_match = True
        else:
            input_match = exp_input == act_input

        is_correct = name_match and input_match
        details.append(
            {
                "step": idx + 1,
                "expected_tool_name": exp_name,
                "actual_tool_name": act_name,
                "name_match": name_match,
                "input_match": input_match,
                "is_correct": is_correct,
            }
        )
        if is_correct:
            matched_steps += 1
        else:
            break

    score = matched_steps / len(expected_calls)
    return {
        "score": score,
        "details": {
            "matched_steps": matched_steps,
            "total_expected_steps": len(expected_calls),
            "call_details": details,
        },
    }
