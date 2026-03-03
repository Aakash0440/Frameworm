# FRAMEWORM-AGENT

Autonomous training monitor for FRAMEWORM. Detects anomalies in
live training runs, identifies root causes, and intervenes
autonomously — adjusting LR, rolling back checkpoints, or alerting.

## Quick Start

```bash
# 1. Add hook to training/trainer.py (4 lines — see docs)
# 2. Install LLM dependency
pip install openai          # or: Ollama for free local LLM

# 3. Set LLM credentials
export FRAMEWORM_AGENT_LLM_API_KEY=your_key
# For free local inference (Ollama):
export FRAMEWORM_AGENT_LLM_BASE_URL=http://localhost:11434/v1
export FRAMEWORM_AGENT_LLM_MODEL=llama3.2

# 4. Start agent alongside your training run
python -m frameworm agent start \
    --config configs/models/gan/dcgan.yaml \
    --model DCGAN \
    --total-steps 50000
```

## Architecture Overview

```
Training Loop (existing)              FRAMEWORM-AGENT (new)
─────────────────────                 ──────────────────────
trainer._train_step()       ──────>   MetricStream (polls W&B or file)
trainer._agent_plugin       <──────   AgentControlPlugin (commands)
                                           │
                                      SignalExtractor (16 features)
                                           │
                                      RuleEngine (fast, no LLM)
                                           │ anomaly detected
                                      BeliefUpdater (POMDP belief)
                                           │
                                      FailurePredictor (proactive)
                                           │
                                      AttributionEngine (causal)
                                           │
                                      FramewormAgent (LLM or CQL policy)
                                           │
                                      AgentControlPlugin (execute)
                                           │
                                      Verifier (KS test, 50 steps)
                                           │
                                      TwinRunner (shadow run)
                                           │
                                      ExperienceBuffer → DB
```

## Configuration

All thresholds in `configs/agent_thresholds.yaml`:

```yaml
agent:
  grad_explosion_threshold: 10.0
  loss_spike_z_score: 3.0
  plateau_min_steps: 100
  cooldown_steps: 200
```

## Training the ML Models

After collecting 10+ experiment runs:

```bash
# Train the gradient forecaster (proactive prediction)
python scripts/train_forecaster.py

# Train the CQL policy (replace LLM for seen anomaly types)
python scripts/train_policy.py
```

## Running the Benchmark

```bash
# Full benchmark (generates paper results)
python scripts/run_benchmark.py

# Specific scenarios
python scripts/run_benchmark.py \
    --scenarios grad_explosion_severe plateau_moderate \
    --baselines RULE_BASED FULL_AGENT

# View results
cat experiments/benchmark/results_table.md
```

## Checking Agent Status

```bash
python -m frameworm agent status --n 10
```

## Test Suite

```bash
# Run all agent tests
pytest tests/agent/ -v

# Individual modules
pytest tests/agent/test_classifier.py -v
pytest tests/agent/test_counterfactual.py -v
pytest tests/agent/test_policy.py -v
```

## The 4-Line Training Hook

In `training/trainer.py`, at end of your `_train_step()`:

```python
# FRAMEWORM-AGENT hook
if hasattr(self, '_agent_plugin') and self._agent_plugin is not None:
    if self.global_step % 50 == 0:
        self._agent_plugin.check_commands()
```

That is the only change to existing code across all 6 parts.

## LLM Options

| Provider  | Cost   | Quality | Setup |
|-----------|--------|---------|-------|
| GPT-4o-mini | $0.15/1M | ★★★★☆ | `pip install openai` |
| Ollama llama3.2 | Free | ★★★☆☆ | [ollama.ai](https://ollama.ai) |
| Ollama qwen2.5 | Free | ★★★★☆ | `ollama pull qwen2.5` |
| Together AI | Low   | ★★★★☆ | Set `BASE_URL` |

## Research Usage

For the paper's counterfactual evaluation:

```python
from agent.counterfactual.delta_tracker import DeltaTracker
from agent.counterfactual.eval_report import EvalReportGenerator

tracker = DeltaTracker()
# (populated automatically by agent during training)

gen = EvalReportGenerator(tracker)
report = gen.generate()
print(report.to_markdown_table())  # paste into paper
```

For the POMDP formalization:

```python
from agent.pomdp.state_space import POMDPSpec
spec = POMDPSpec()
print(spec.to_paper_text())  # paste into Section 3
```
