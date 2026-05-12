#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Короткая CLI-обёртка для пошаговой оценки одного запуска."""

from __future__ import annotations

import argparse
from pathlib import Path

from step_by_step import evaluate_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Пошаговая оценка одного run-dir")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Директория запуска внутри evaluate_langchain, например evaluate_langchain/gpt-4o_dev_IF_xx-xx-xx_xx-xx",
    )
    parser.add_argument("--benchmark", type=Path, default=Path("./benchmark/development_quest.json"))
    args = parser.parse_args()

    results = evaluate_run_dir(args.run_dir, args.benchmark)
    print(f"Saved: {args.run_dir / 'step_by_step_evaluation_results.json'}")
    print(f"Evaluated: {results['summary']['evaluated_questions']} / {results['summary']['total_questions']}")


if __name__ == "__main__":
    main()
