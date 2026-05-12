#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Пошаговая (trajectory) оценка для development-бенчмарка.

Скрипт сравнивает фактические tool-calls агента с эталонными цепочками из benchmark:
- источник факта: `evaluate_langchain/<run_dir>/gpt-4o_IF_langchain.json`
  (или уже подготовленный `extracted_tool_calls.json`);
- источник ожиданий: `benchmark/development_quest.json`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from dev_eval_utils import (
    contains_all_tool_calls_any_order,
    contains_all_tool_calls_in_order,
    dump_json,
    extract_tool_calls_from_langchain_log,
    find_single_log_file,
    load_dev_ground_truth,
    load_json,
    normalize_question_index,
    parameter_accuracy,
    trajectory_step_wise_score,
)


def _best_metric_across_candidates(metric_fn, actual_calls: List[Dict[str, Any]], candidates: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Выбирает лучший score среди нескольких допустимых эталонных цепочек (`any_of`).

    Почему так:
    - в multi-agent вопросах заранее допускаются несколько корректных траекторий;
    - оцениваем не по одной фиксированной цепочке, а по лучшему совпадению.
    """
    if not candidates:
        candidates = [[]]
    best_result: Dict[str, Any] | None = None
    best_idx = 0
    for idx, candidate in enumerate(candidates):
        result = metric_fn(actual_calls, candidate)
        if best_result is None or result.get("score", 0.0) > best_result.get("score", 0.0):
            best_result = result
            best_idx = idx
    assert best_result is not None
    details = dict(best_result.get("details", {}))
    details["matched_candidate_index"] = best_idx
    return {"score": best_result.get("score", 0.0), "details": details}


def _index_predicted_calls(predicted_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Индексирует предсказания по `question_index` для O(1) доступа в цикле оценки."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in predicted_rows:
        qidx = normalize_question_index(row.get("question_index", ""))
        out[qidx] = row
    return out


def run_step_by_step_evaluation(
    predicted_file: Path,
    benchmark_file: Path,
) -> Dict[str, Any]:
    """Основной расчёт step-by-step метрик.

    Вход:
    - `predicted_file`: `extracted_tool_calls.json` (нормализованные фактические вызовы),
    - `benchmark_file`: `development_quest.json` (ground truth).
    """
    gt = load_dev_ground_truth(benchmark_file)
    predicted_rows = load_json(predicted_file)
    if not isinstance(predicted_rows, list):
        raise ValueError(f"Expected list in {predicted_file}")

    pred = _index_predicted_calls(predicted_rows)

    summary = {
        "total_questions": len(gt),
        "evaluated_questions": 0,
        "missing_predictions": [],
        "metrics_summary": {
            "contains_all_tool_calls_any_order": {"total_score": 0.0, "count": 0},
            "contains_all_tool_calls_in_order": {"total_score": 0.0, "count": 0},
            "trajectory_step_wise_score": {"total_score": 0.0, "count": 0},
            "parameter_accuracy": {"total_score": 0.0, "count": 0},
        },
    }
    details: Dict[str, Any] = {}

    for qidx in sorted(gt.keys(), key=lambda x: int(x.replace("question", "")) if x.replace("question", "").isdigit() else x):
        gt_item = gt[qidx]
        if qidx not in pred:
            summary["missing_predictions"].append(qidx)
            continue

        summary["evaluated_questions"] += 1
        actual_calls = pred[qidx].get("tool_calls", [])
        expected_calls = gt_item.get("expected_tool_calls", [])
        expected_multi = gt_item.get("expected_multiagent_chains", {})

        # Мультиагентный режим: считаем метрики отдельно по каждому агенту,
        # затем усредняем по агентам, помеченным как обязательные.
        if isinstance(expected_multi, dict) and expected_multi:
            actual_by_agent = pred[qidx].get("tool_calls_by_agent", {})
            metric_fns = {
                "contains_all_tool_calls_any_order": contains_all_tool_calls_any_order,
                "contains_all_tool_calls_in_order": contains_all_tool_calls_in_order,
                "trajectory_step_wise_score": trajectory_step_wise_score,
                "parameter_accuracy": parameter_accuracy,
            }

            agent_metric_details: Dict[str, Any] = {}
            metric_scores: Dict[str, List[float]] = {k: [] for k in metric_fns.keys()}

            for agent_key, spec in expected_multi.items():
                if not isinstance(spec, dict):
                    continue
                required = bool(spec.get("required", True))
                candidates = spec.get("any_of", [])
                if not isinstance(candidates, list):
                    candidates = []
                actual_agent_calls = actual_by_agent.get(agent_key, [])

                per_agent_results: Dict[str, Any] = {}
                for metric_name, metric_fn in metric_fns.items():
                    best = _best_metric_across_candidates(metric_fn, actual_agent_calls, candidates)
                    per_agent_results[metric_name] = best
                    if required:
                        metric_scores[metric_name].append(best["score"])

                agent_metric_details[agent_key] = {
                    "required": required,
                    "actual_tool_names": [c.get("name", "") for c in actual_agent_calls],
                    "candidate_count": len(candidates),
                    "metrics": per_agent_results,
                }

            q_result = {}
            for metric_name in metric_fns.keys():
                scores = metric_scores.get(metric_name, [])
                avg_score = (sum(scores) / len(scores)) if scores else 1.0
                q_result[metric_name] = {
                    "score": avg_score,
                    "details": {
                        "mode": "multiagent",
                        "agent_scores": {
                            agent_key: agent_metric_details[agent_key]["metrics"][metric_name]["score"]
                            for agent_key in agent_metric_details.keys()
                        },
                        "agent_metric_details": {
                            agent_key: agent_metric_details[agent_key]["metrics"][metric_name]
                            for agent_key in agent_metric_details.keys()
                        },
                    },
                }
        else:
            # Одноагентный путь: сравнение одной общей цепочки.
            q_result = {
                "contains_all_tool_calls_any_order": contains_all_tool_calls_any_order(actual_calls, expected_calls),
                "contains_all_tool_calls_in_order": contains_all_tool_calls_in_order(actual_calls, expected_calls),
                "trajectory_step_wise_score": trajectory_step_wise_score(actual_calls, expected_calls),
                "parameter_accuracy": parameter_accuracy(actual_calls, expected_calls),
            }
        details[qidx] = q_result

        for metric_name, metric_res in q_result.items():
            summary["metrics_summary"][metric_name]["total_score"] += metric_res["score"]
            summary["metrics_summary"][metric_name]["count"] += 1

    for metric_name, metric_stats in summary["metrics_summary"].items():
        if metric_stats["count"]:
            metric_stats["average_score"] = metric_stats["total_score"] / metric_stats["count"]
        else:
            metric_stats["average_score"] = 0.0

    return {"individual_results": details, "summary": summary}


def evaluate_run_dir(run_dir: Path, benchmark_file: Path) -> Dict[str, Any]:
    """Оценивает один run-dir и сохраняет `step_by_step_evaluation_results.json`."""
    extracted_path = run_dir / "extracted_tool_calls.json"
    if not extracted_path.exists():
        # Если нормализованный слой вызовов ещё не создан, строим его из сырого лога диалога.
        log_path = find_single_log_file(run_dir)
        extracted = extract_tool_calls_from_langchain_log(log_path)
        dump_json(extracted_path, extracted)

    results = run_step_by_step_evaluation(extracted_path, benchmark_file)
    output_path = run_dir / "step_by_step_evaluation_results.json"
    dump_json(output_path, results)
    return results


def evaluate_batch(root_dir: Path, benchmark_file: Path) -> Dict[str, Any]:
    """Пакетная оценка всех запусков в директории `root_dir`."""
    batch: Dict[str, Any] = {}
    for run_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        try:
            batch[run_dir.name] = evaluate_run_dir(run_dir, benchmark_file)
        except Exception as exc:
            batch[run_dir.name] = {"error": str(exc)}
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Пошаговая оценка development-бенчмарка")
    parser.add_argument("--run-dir", type=Path, help="Один run-dir внутри evaluate_langchain")
    parser.add_argument("--root-dir", type=Path, default=Path("./evaluate_langchain"), help="Корневая папка для batch-оценки")
    parser.add_argument("--benchmark", type=Path, default=Path("./benchmark/development_quest.json"), help="Путь к benchmark JSON")
    args = parser.parse_args()

    if args.run_dir:
        results = evaluate_run_dir(args.run_dir, args.benchmark)
        print(f"Saved: {args.run_dir / 'step_by_step_evaluation_results.json'}")
        print(f"Evaluated: {results['summary']['evaluated_questions']} / {results['summary']['total_questions']}")
        return

    batch = evaluate_batch(args.root_dir, args.benchmark)
    output_file = Path("./evaluate/batch_step_by_step_results.json")
    dump_json(output_file, batch)
    print(f"Saved batch results: {output_file}")


if __name__ == "__main__":
    main()
