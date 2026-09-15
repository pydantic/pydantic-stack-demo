"""Evaluate, compare and optimize the contact-extraction prompt.

uv run prompt-optimization eval [--expert]         # score one prompt on the dataset
uv run prompt-optimization compare                 # initial vs expert prompt, side by side
uv run prompt-optimization optimize --max-calls 20 [--output FILE]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import cast

import logfire
from gepa.api import optimize

from demo_core import configure_logfire, settings
from prompt_optimization.adapter import EvalsGEPAAdapter
from prompt_optimization.evals import contact_dataset
from prompt_optimization.task import EXPERT_INSTRUCTIONS, INITIAL_INSTRUCTIONS, contact_agent, extract_contact_info


def run_evaluation(instructions: str, *, label: str) -> float:
    print(f'\n== {label} ==\n{instructions[:100]}{"..." if len(instructions) > 100 else ""}')

    async def evaluate():
        with contact_agent.override(instructions=instructions):
            return await contact_dataset.evaluate(
                extract_contact_info, name=f'contact extraction: {label}', max_concurrency=5, progress=False
            )

    report = asyncio.run(evaluate())
    report.print(include_input=False, include_output=False)
    if url := logfire.url_from_eval(report):
        print(f'Logfire experiment: {url}')

    accuracies = [float(c.scores['accuracy'].value) for c in report.cases if 'accuracy' in c.scores]
    average = sum(accuracies) / len(accuracies) if accuracies else 0.0
    print(f'Average accuracy: {average:.1%}')
    return average


def run_optimization(max_metric_calls: int, output_file: Path | None) -> str:
    print(f'\nOptimizing with GEPA, at most {max_metric_calls} evaluation calls...')
    adapter = EvalsGEPAAdapter(dataset=contact_dataset, task=extract_contact_info, agent=contact_agent)

    result = optimize(
        seed_candidate={'instructions': json.dumps(INITIAL_INSTRUCTIONS)},
        trainset=contact_dataset.cases,
        valset=contact_dataset.cases,  # one small dataset for both; use a held-out valset for real work
        adapter=adapter,
        max_metric_calls=max_metric_calls,
        display_progress_bar=True,
    )

    best = json.loads(cast(dict[str, str], result.best_candidate)['instructions'])
    print(f'\nBest validation score: {result.val_aggregate_scores[result.best_idx]:.1%}')
    print(f'\nOptimized instructions:\n{best}')
    if output_file:
        output_file.write_text(best)
        print(f'\nSaved to {output_file}')
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description='Prompt optimization with GEPA, Pydantic AI and Pydantic Evals')
    sub = parser.add_subparsers(dest='command', required=True)
    eval_parser = sub.add_parser('eval', help='Evaluate one prompt')
    eval_parser.add_argument('--expert', action='store_true', help='Use the expert prompt instead of the initial one')
    sub.add_parser('compare', help='Evaluate the initial and expert prompts')
    opt_parser = sub.add_parser('optimize', help='Run GEPA optimization')
    opt_parser.add_argument('--max-calls', type=int, default=20, help='Maximum evaluation calls (default: 20)')
    opt_parser.add_argument('--output', type=Path, help='Write the optimized prompt to this file')
    args = parser.parse_args()

    configure_logfire('prompt-optimization')
    settings().require_model_credentials()

    if args.command == 'eval':
        if args.expert:
            run_evaluation(EXPERT_INSTRUCTIONS, label='expert')
        else:
            run_evaluation(INITIAL_INSTRUCTIONS, label='initial')
    elif args.command == 'compare':
        run_evaluation(INITIAL_INSTRUCTIONS, label='initial')
        run_evaluation(EXPERT_INSTRUCTIONS, label='expert')
    else:
        run_optimization(args.max_calls, args.output)


if __name__ == '__main__':
    main()
