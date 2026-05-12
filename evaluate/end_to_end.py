#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end оценка для development-бенчмарка.

Скрипт отвечает на два вопроса:
1) correctness: насколько часто финальный ответ совпадает с эталоном;
2) efficiency: насколько длинной была фактическая tool-траектория относительно эталона.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from dev_eval_utils import (
    denormalize_question_id,
    dump_json,
    extract_answer_letter,
    extract_tool_calls_from_langchain_log,
    find_single_log_file,
    load_dev_ground_truth,
    load_json,
    normalize_question_index,
    polish_results_summary,
    get_expected_tool_count,
)


def _index_predicted_answers(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Индексирует ответы модели по вопросу для быстрого сопоставления с ground truth."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qid = row.get("question_id", "")
        qidx = normalize_question_index(qid)
        out[qidx] = row
    return out


def _index_predicted_tools(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Индексирует extracted tool-calls по question_index."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qidx = normalize_question_index(row.get("question_index", ""))
        out[qidx] = row
    return out


def run_end_to_end_evaluation(
    benchmark_file: Path,
    predicted_answers_file: Path,
    predicted_tool_calls_file: Path,
) -> Dict[str, Any]:
    """Считает end-to-end метрики на одном запуске.

    Источники:
    - benchmark (`development_quest.json`) -> эталон ответов и ожидаемый baseline по шагам;
    - `results_summary_polished.json` -> нормализованные ответы модели;
    - `extracted_tool_calls.json` -> фактическое число tool-вызовов.
    """
    gt = load_dev_ground_truth(benchmark_file)
    predicted_answers = load_json(predicted_answers_file)
    predicted_tools = load_json(predicted_tool_calls_file)
    if not isinstance(predicted_answers, list):
        raise ValueError(f"Expected list in {predicted_answers_file}")
    if not isinstance(predicted_tools, list):
        raise ValueError(f"Expected list in {predicted_tool_calls_file}")

    answers_idx = _index_predicted_answers(predicted_answers)
    tools_idx = _index_predicted_tools(predicted_tools)

    accuracy = {
        "total_questions": len(gt),
        "evaluated_questions": 0,
        "correct_answers": 0,
        "fail_answers": 0,
        "unknown_answers": 0,
        "missing_predictions": [],
        "accuracy": 0.0,
        "detailed_results": [],
    }
    efficiency = {
        "total_questions": len(gt),
        "evaluated_questions": 0,
        "efficiency_scores": [],
        "average_efficiency": 0.0,
        "detailed_results": [],
    }

    for qidx in sorted(gt.keys(), key=lambda x: int(x.replace("question", "")) if x.replace("question", "").isdigit() else x):
        gt_answer = gt[qidx]["final_answer"]
        expected_tool_count = get_expected_tool_count(gt[qidx])

        # Точность: сравнение финальной буквы ответа с эталонной буквой из benchmark.
        if qidx not in answers_idx:
            accuracy["missing_predictions"].append(qidx)
            accuracy["detailed_results"].append(
                {
                    "question_index": qidx,
                    "ground_truth": gt_answer,
                    "predicted": "MISSING",
                    "correct": False,
                    "status": "missing",
                }
            )
        else:
            accuracy["evaluated_questions"] += 1
            pred_row = answers_idx[qidx]
            pred_text = pred_row.get("final_answer", "")
            pred_letter = extract_answer_letter(pred_text)
            is_correct = pred_letter == gt_answer

            if pred_letter == "FAIL":
                accuracy["fail_answers"] += 1
                status = "fail"
            elif pred_letter == "UNKNOWN":
                accuracy["unknown_answers"] += 1
                status = "unknown"
            elif is_correct:
                accuracy["correct_answers"] += 1
                status = "correct"
            else:
                status = "incorrect"

            accuracy["detailed_results"].append(
                {
                    "question_index": qidx,
                    "ground_truth": gt_answer,
                    "predicted": pred_letter,
                    "correct": is_correct,
                    "status": status,
                }
            )

        # Эффективность: фактические шаги / ожидаемые шаги.
        # Значение > 1.0 означает "агент сделал больше шагов, чем эталонный минимум".
        if qidx not in tools_idx:
            efficiency["detailed_results"].append(
                {
                    "question_index": qidx,
                    "gt_tool_count": expected_tool_count,
                    "model_tool_count": 0,
                    "efficiency": 0.0,
                    "status": "missing",
                }
            )
        else:
            efficiency["evaluated_questions"] += 1
            model_tool_count = len(tools_idx[qidx].get("tool_calls", []))
            if expected_tool_count == 0:
                eff = 1.0 if model_tool_count == 0 else float("inf")
            else:
                eff = model_tool_count / expected_tool_count

            efficiency["efficiency_scores"].append(eff)
            efficiency["detailed_results"].append(
                {
                    "question_index": qidx,
                    "gt_tool_count": expected_tool_count,
                    "model_tool_count": model_tool_count,
                    "efficiency": eff,
                    "status": "evaluated",
                }
            )

    if accuracy["evaluated_questions"] > 0:
        accuracy["accuracy"] = accuracy["correct_answers"] / accuracy["evaluated_questions"]

    if efficiency["efficiency_scores"]:
        efficiency["average_efficiency"] = sum(efficiency["efficiency_scores"]) / len(efficiency["efficiency_scores"])

    return {
        "accuracy": accuracy,
        "efficiency": efficiency,
        "summary": {
            "total_questions": accuracy["total_questions"],
            "accuracy_rate": accuracy["accuracy"],
            "average_efficiency": efficiency["average_efficiency"],
            "fail_rate": (
                accuracy["fail_answers"] / accuracy["evaluated_questions"] if accuracy["evaluated_questions"] else 0.0
            ),
        },
    }


def evaluate_run_dir(run_dir: Path, benchmark_file: Path) -> Dict[str, Any]:
    """Оценивает один run-dir и пишет `end_to_end_evaluation_results.json`."""
    extracted_path = run_dir / "extracted_tool_calls.json"
    if not extracted_path.exists():
        # Автоматически строим нормализованный слой вызовов, если его нет.
        log_path = find_single_log_file(run_dir)
        extracted = extract_tool_calls_from_langchain_log(log_path)
        dump_json(extracted_path, extracted)

    polished_path = run_dir / "results_summary_polished.json"
    if not polished_path.exists():
        # Автоматически нормализуем ответы, чтобы метрика точности работала стабильно.
        raw_results = run_dir / "results_summary.json"
        polished = polish_results_summary(raw_results)
        dump_json(polished_path, polished)

    results = run_end_to_end_evaluation(
        benchmark_file=benchmark_file,
        predicted_answers_file=polished_path,
        predicted_tool_calls_file=extracted_path,
    )
    output_path = run_dir / "end_to_end_evaluation_results.json"
    dump_json(output_path, results)
    return results


def evaluate_batch(root_dir: Path, benchmark_file: Path) -> Dict[str, Any]:
    """Пакетная оценка всех run-директорий в `root_dir`."""
    batch: Dict[str, Any] = {}
    for run_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        try:
            batch[run_dir.name] = evaluate_run_dir(run_dir, benchmark_file)
        except Exception as exc:
            batch[run_dir.name] = {"error": str(exc)}
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end оценка development-бенчмарка")
    parser.add_argument("--run-dir", type=Path, help="Один run-dir внутри evaluate_langchain")
    parser.add_argument("--root-dir", type=Path, default=Path("./evaluate_langchain"), help="Корневая папка для batch-оценки")
    parser.add_argument("--benchmark", type=Path, default=Path("./benchmark/development_quest.json"), help="Путь к benchmark JSON")
    args = parser.parse_args()

    if args.run_dir:
        results = evaluate_run_dir(args.run_dir, args.benchmark)
        print(f"Saved: {args.run_dir / 'end_to_end_evaluation_results.json'}")
        print(f"Accuracy: {results['summary']['accuracy_rate']:.4f}")
        print(f"Avg efficiency: {results['summary']['average_efficiency']:.4f}")
        return

    batch = evaluate_batch(args.root_dir, args.benchmark)
    output_file = Path("./evaluate/batch_evaluation_results.json")
    dump_json(output_file, batch)
    print(f"Saved batch results: {output_file}")


if __name__ == "__main__":
    main()
