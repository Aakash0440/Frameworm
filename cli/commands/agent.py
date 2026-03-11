"""
CLI commands for FRAMEWORM-AGENT.

Adds to your existing FRAMEWORM CLI:
    frameworm agent start   → start agent alongside a training run
    frameworm agent status  → show agent status and recent decisions
    frameworm agent train   → train forecaster + CQL policy
    frameworm agent bench   → run benchmark suite
    frameworm agent report  → generate eval report from logged deltas

Usage:
    python -m frameworm agent start --run-id abc123 --config configs/dcgan.yaml
    python -m frameworm agent status
    python -m frameworm agent train --forecaster --policy
    python -m frameworm agent bench --scenarios all --baselines RULE_BASED FULL_AGENT
    python -m frameworm agent report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def add_agent_parser(subparsers) -> None:
    """
    Register 'agent' subcommand with your existing CLI parser.

    In your cli/main.py or cli/cli.py:
        from cli.commands.agent import add_agent_parser
        add_agent_parser(subparsers)
    """
    agent_parser = subparsers.add_parser(
        "agent",
        help="FRAMEWORM-AGENT: autonomous training monitor",
    )
    agent_sub = agent_parser.add_subparsers(dest="agent_command", required=True)

    # ── start ──────────────────────────────────────────────────────
    start_p = agent_sub.add_parser("start", help="Start agent daemon")
    start_p.add_argument(
        "--run-id", type=str, default=None, help="W&B run ID to monitor (omit for local mode)"
    )
    start_p.add_argument(
        "--config", type=str, default="configs/base.yaml", help="FRAMEWORM config file"
    )
    start_p.add_argument(
        "--model", type=str, default="unknown", help="Model architecture name (e.g. DCGAN)"
    )
    start_p.add_argument("--total-steps", type=int, default=10_000, help="Total training steps")
    start_p.add_argument(
        "--block", action="store_true", help="Block until training ends (default: daemon)"
    )

    # ── status ─────────────────────────────────────────────────────
    status_p = agent_sub.add_parser("status", help="Show agent status")
    status_p.add_argument(
        "--log-dir", type=str, default="experiments/agent_logs", help="Agent log directory"
    )
    status_p.add_argument("--n", type=int, default=5, help="Last N decisions to show")

    # ── train ──────────────────────────────────────────────────────
    train_p = agent_sub.add_parser("train", help="Train forecaster and/or policy")
    train_p.add_argument("--forecaster", action="store_true", help="Train GradForecaster LSTM")
    train_p.add_argument("--policy", action="store_true", help="Train CQL policy")
    train_p.add_argument("--epochs", type=int, default=50)

    # ── bench ──────────────────────────────────────────────────────
    bench_p = agent_sub.add_parser("bench", help="Run benchmark suite")
    bench_p.add_argument(
        "--scenarios", nargs="*", default=None, help="Scenario names (default: all 12)"
    )
    bench_p.add_argument(
        "--baselines",
        nargs="*",
        default=["HUMAN", "RULE_BASED", "LLM_ONLY", "FULL_AGENT"],
        help="Baselines to compare",
    )
    bench_p.add_argument("--steps", type=int, default=800, help="Steps per scenario")
    bench_p.add_argument("--seed", type=int, default=42)

    # ── report ─────────────────────────────────────────────────────
    report_p = agent_sub.add_parser("report", help="Generate eval report")
    report_p.add_argument(
        "--delta-dir", type=str, default="experiments/agent_logs/deltas", help="Delta log directory"
    )
    report_p.add_argument(
        "--out", type=str, default="experiments/agent_logs", help="Output directory"
    )


def run_agent_command(args) -> int:
    """
    Dispatch agent subcommand.
    Called from your cli/main.py after parsing args.

    Returns exit code (0 = success).
    """
    cmd = args.agent_command

    if cmd == "start":
        return _cmd_start(args)
    elif cmd == "status":
        return _cmd_status(args)
    elif cmd == "train":
        return _cmd_train(args)
    elif cmd == "bench":
        return _cmd_bench(args)
    elif cmd == "report":
        return _cmd_report(args)
    else:
        print(f"Unknown agent command: {cmd}")
        return 1


def _cmd_start(args) -> int:
    """Start the agent daemon."""
    print(f"Starting FRAMEWORM-AGENT...")
    print(f"  Config:      {args.config}")
    print(f"  Model:       {args.model}")
    print(f"  Total steps: {args.total_steps:,}")
    print(f"  Run ID:      {args.run_id or 'LOCAL MODE'}")
    print()

    try:
        from agent.react.agent import FramewormAgent

        agent = FramewormAgent.from_config(
            config_path=args.config,
            run_id=args.run_id,
            model_name=args.model,
            total_steps=args.total_steps,
        )

        if args.block:
            print("Agent running (blocking mode)...")
            agent.run()
        else:
            agent.start()
            print("Agent started (daemon mode). Training can now begin.")
            print("  Stop: Ctrl+C or agent.stop()")

        return 0

    except Exception as exc:
        print(f"Error starting agent: {exc}")
        logger.exception("agent start failed")
        return 1


def _cmd_status(args) -> int:
    """Show recent agent decisions."""
    log_dir = Path(args.log_dir)
    log_files = sorted(log_dir.glob("agent_decisions_*.json"))

    if not log_files:
        print(f"No decision logs found in {log_dir}")
        print("Run a training job with the agent first.")
        return 0

    latest = log_files[-1]
    print(f"Latest log: {latest}")
    print()

    try:
        with open(latest) as f:
            decisions = json.load(f)

        recent = decisions[-args.n :]
        print(f"Last {len(recent)} decisions:\n")
        for d in recent:
            resolved = "✓" if d.get("resolved") else "✗"
            print(
                f"  Step {d.get('step', '?'):>6,} | "
                f"{d.get('anomaly_type', '?'):25s} | "
                f"{d.get('action', '?'):20s} | "
                f"{resolved} resolved | "
                f"loss_Δ={d.get('loss_delta', 0.0):+.4f}"
            )

        total = len(decisions)
        resolved_count = sum(1 for d in decisions if d.get("resolved"))
        print(f"\nTotal: {total} decisions, " f"{resolved_count/total:.1%} resolved")
        return 0

    except Exception as exc:
        print(f"Error reading log: {exc}")
        return 1


def _cmd_train(args) -> int:
    """Train forecaster and/or CQL policy."""
    if not args.forecaster and not args.policy:
        print("Specify --forecaster, --policy, or both.")
        return 1

    if args.forecaster:
        print("Training GradForecaster LSTM...")
        try:
            from agent.forecaster.training_data import DataCollector, ForecasterDataset
            from agent.forecaster.grad_forecaster import (
                GradForecaster,
                ForecasterConfig,
                train_forecaster,
            )

            collector = DataCollector()
            samples = collector.collect_all()
            print(f"  Collected {len(samples)} samples from experiments/")

            if len(samples) < 10:
                print("  Not enough data yet (need 10+). " "Run more training experiments first.")
            else:
                dataset = ForecasterDataset(samples)
                config = ForecasterConfig(max_epochs=args.epochs)
                forecaster = GradForecaster(config)
                history = train_forecaster(forecaster, dataset)
                best = min(history.get("val_loss", [float("inf")]))
                print(f"  Done. Best val_loss: {best:.4f}")
                print(f"  Saved: experiments/forecaster/best_forecaster.pt")

        except Exception as exc:
            print(f"  Forecaster training failed: {exc}")
            return 1

    if args.policy:
        print("\nTraining CQL policy...")
        try:
            from agent.policy.experience_buffer import ExperienceBuffer
            from agent.policy.cql_policy import CQLPolicy, CQLConfig, train_cql_policy
            from agent.policy.policy_eval import PolicyEvaluator

            buffer = ExperienceBuffer()
            n = buffer.load_from_db()
            print(f"  Loaded {n} transitions from DB")

            if not buffer.is_ready:
                print(
                    f"  Not enough transitions (need 100, have {len(buffer)}). "
                    "Run more training experiments first."
                )
            else:
                config = CQLConfig(max_epochs=args.epochs)
                policy = CQLPolicy(config=config)
                history = train_cql_policy(policy, buffer)

                evaluator = PolicyEvaluator(policy, buffer)
                result = evaluator.evaluate()
                print(f"  Done. Win rate vs LLM: {result.overall_win_rate:.1%}")
                evaluator.save_learning_curve()
                print(f"  Saved: experiments/policy/best_cql_policy.pt")
                print(f"  Saved: experiments/policy/learning_curve.json")

        except Exception as exc:
            print(f"  Policy training failed: {exc}")
            return 1

    return 0


def _cmd_bench(args) -> int:
    """Run benchmark suite."""
    print("Running FRAMEWORM-AGENT benchmark suite...")
    print(f"  Scenarios: {args.scenarios or 'all 12'}")
    print(f"  Baselines: {args.baselines}")
    print(f"  Steps/scenario: {args.steps}")
    print()

    try:
        from agent.benchmark.run_suite import BenchmarkSuite
        from agent.benchmark.results_analyzer import ResultsAnalyzer

        suite = BenchmarkSuite(seed=args.seed)
        result = suite.run(
            scenario_names=args.scenarios,
            baselines=args.baselines,
            n_steps_per_scenario=args.steps,
        )

        analyzer = ResultsAnalyzer(result)
        analyzer.save()

        print(
            f"\nBenchmark complete: {len(result.results)} runs, "
            f"{result.total_duration_seconds:.1f}s"
        )
        return 0

    except Exception as exc:
        print(f"Benchmark failed: {exc}")
        logger.exception("bench failed")
        return 1


def _cmd_report(args) -> int:
    """Generate evaluation report from logged deltas."""
    print("Generating evaluation report...")

    try:
        from agent.counterfactual.delta_tracker import DeltaTracker
        from agent.counterfactual.eval_report import EvalReportGenerator

        delta_dir = Path(args.delta_dir)
        if not delta_dir.exists():
            print(f"Delta directory not found: {delta_dir}")
            return 1

        tracker = DeltaTracker(log_dir=delta_dir)
        gen = EvalReportGenerator(tracker, log_dir=Path(args.out))
        report = gen.generate()

        print(f"\nEval Report:")
        print(f"  Total interventions:    {report.n_total_interventions}")
        print(f"  With shadow:            {report.n_with_shadow}")
        print(f"  Overall success rate:   {report.overall_success_rate:.1%}")
        print(f"  Mean loss delta:        {report.overall_mean_loss_delta:+.4f}")
        print(f"  Welch p-value:          {report.overall_welch_p:.4f}")
        print(f"  Statistically sig.:     {report.is_overall_significant}")
        print(f"\n  Saved to: {args.out}/")
        return 0

    except Exception as exc:
        print(f"Report generation failed: {exc}")
        return 1
