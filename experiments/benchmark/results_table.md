# FRAMEWORM-AGENT Benchmark Results

## Overall Performance

| Baseline | Detection Rate | Resolution Rate | Mean Detection Latency | Mean Recovery Steps |
|---|---|---|---|---|
| HUMAN | 0.0% | 0.0% | N/A steps | N/A steps |
| RULE_BASED | 50.0% | 0.0% | 0 steps | N/A steps |
| LLM_ONLY | 0.0% | 0.0% | N/A steps | N/A steps |
| FULL_AGENT | 100.0% | 50.0% | 0 steps | 100 steps |

## Per Anomaly Type — Resolution Rate

| Anomaly Type | HUMAN | RULE_BASED | LLM_ONLY | FULL_AGENT |
|---|---|---|---|---|
| GRADIENT_EXPLOSION | 0.0% | 0.0% | 0.0% | 100.0% |
| LOSS_SPIKE | 0.0% | 0.0% | 0.0% | 0.0% |
| PLATEAU | 0.0% | 0.0% | 0.0% | 0.0% |
| DIVERGENCE | 0.0% | 0.0% | 0.0% | 0.0% |

## Per Severity — Detection Rate

| Severity | HUMAN | RULE_BASED | LLM_ONLY | FULL_AGENT |
|---|---|---|---|---|
| MILD | 0.0% | 0.0% | 0.0% | 0.0% |
| MODERATE | 0.0% | 100.0% | 0.0% | 100.0% |
| SEVERE | 0.0% | 0.0% | 0.0% | 100.0% |

*Generated from 4 benchmark runs. Total duration: 0.1s*