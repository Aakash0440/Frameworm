"""
Full test runner for DEPLOY steps 6–10.
Run with: python test_deploy_steps6_10.py
No pytest required.
"""

import sys

sys.path.insert(0, ".")

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
all_results = []


def run(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        all_results.append(True)
    except Exception as e:
        print(f"  {FAIL} {name}  —  {e}")
        all_results.append(False)


print("\n══════════════════════════════════════════════════")
print("  FRAMEWORM DEPLOY — Steps 6–10 test suite")
print("══════════════════════════════════════════════════\n")

from deploy.tests.test_export import (test_docker_compose_generated,
                                      test_dockerfile_generated,
                                      test_server_builder_all_model_types,
                                      test_server_builder_generates_file,
                                      test_server_builder_shift_injection)
from deploy.tests.test_registry import (test_full_lifecycle,
                                        test_list_all_doesnt_crash,
                                        test_promote_to_production,
                                        test_register_and_retrieve)
from deploy.tests.test_serving import (
    test_degradation_monitor_no_trigger_on_healthy,
    test_degradation_monitor_triggers, test_health_checker_error_rate,
    test_health_checker_initial_state, test_health_checker_mark_ready)

print("── Serving Layer (Step 6) ────────────────────────")
run("health checker initial state", test_health_checker_initial_state)
run("health checker mark ready", test_health_checker_mark_ready)
run("error rate calculation", test_health_checker_error_rate)

print("\n── Server Builder (Step 7) ───────────────────────")
run("server.py generated correctly", test_server_builder_generates_file)
run("all 6 model types generate", test_server_builder_all_model_types)
run("SHIFT reference injected", test_server_builder_shift_injection)

print("\n── Docker Builder (Step 8) ───────────────────────")
run("Dockerfile generated with HEALTHCHECK", test_dockerfile_generated)
run("docker-compose.yml generated", test_docker_compose_generated)

print("\n── Rollback Monitor (Step 9) ─────────────────────")
run("degradation triggers on high latency", test_degradation_monitor_triggers)
run("no trigger on healthy traffic", test_degradation_monitor_no_trigger_on_healthy)

print("\n── Registry (Steps 1–5 regression) ──────────────")
run("register and retrieve", test_register_and_retrieve)
run("promote to production", test_promote_to_production)
run("full dev→staging→production lifecycle", test_full_lifecycle)
run("list_all doesn't crash", test_list_all_doesnt_crash)

print("\n══════════════════════════════════════════════════")
passed = sum(all_results)
total = len(all_results)
colour = "\033[92m" if passed == total else "\033[91m"
print(f"  {colour}{passed}/{total} passed\033[0m")
print("══════════════════════════════════════════════════\n")
if passed < total:
    sys.exit(1)
