import subprocess, sys, time

for i in range(10):
    print(f"\n=== Experiment {i+1}/10 ===")
    subprocess.run([
        sys.executable, "cli/main.py", "train",
        "--config", "configs/models/gan/dcgan.yaml"
    ])
    time.sleep(2)

print("\nAll experiments done. Ready to train forecaster and policy.")
