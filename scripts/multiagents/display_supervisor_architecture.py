#!/usr/bin/env python3
"""Display MultiAgentSupervisor architecture using native LangGraph methods only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock


# Allow importing `multiagents.Supervisor` when running from repo root.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from multiagents.Supervisor import MultiAgentSupervisor  # noqa: E402


def build_supervisor() -> MultiAgentSupervisor:
    """Initialize supervisor with mocks for visualization only."""
    return MultiAgentSupervisor(
        llm=MagicMock(name="llm"),
        analyze_question_agent=MagicMock(name="analyze_question_agent"),
        location_agent=MagicMock(name="location_agent"),
        data_acquisition_agent=MagicMock(name="data_acquisition_agent"),
        main_agent_executor=MagicMock(name="main_agent_executor"),
        system_prompt="",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Display MultiAgentSupervisor graph")
    parser.add_argument(
        "--format",
        choices=["mermaid", "ascii"],
        default="mermaid",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for text format",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=None,
        help="Optional PNG output path via draw_mermaid_png()",
    )
    args = parser.parse_args()

    supervisor = build_supervisor()
    drawable = supervisor.graph.get_graph()

    if args.format == "mermaid":
        text = drawable.draw_mermaid()
    else:
        text = str(drawable)

    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Saved text graph to: {args.output}")
    else:
        print(text)

    if args.png:
        args.png.write_bytes(drawable.draw_mermaid_png())
        print(f"Saved PNG graph to: {args.png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
