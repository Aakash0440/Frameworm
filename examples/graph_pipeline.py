"""
Example: Using Graph-Based Pipeline

Demonstrates building complex workflows with dependencies.
"""

from pipelines.base import GraphPipeline
from core import Config
import time


def load_data():
    """Simulate data loading"""
    print("Loading data...")
    time.sleep(0.1)
    return {"train": [1, 2, 3], "val": [4, 5]}


def preprocess(data):
    """Simulate preprocessing"""
    print("Preprocessing data...")
    time.sleep(0.1)
    return {k: [x * 2 for x in v] for k, v in data.items()}


def train_model(data):
    """Simulate training"""
    print("Training model...")
    time.sleep(0.2)
    return {"accuracy": 0.95, "loss": 0.05}


def evaluate_model(data, model):
    """Simulate evaluation"""
    print("Evaluating model...")
    time.sleep(0.1)
    return {"test_accuracy": 0.93}


def visualize(metrics, eval_results):
    """Simulate visualization"""
    print("Creating visualizations...")
    time.sleep(0.1)
    return {"plot": "results.png"}


def main():
    print("Graph Pipeline Example")
    print("=" * 60)
    
    # Create pipeline
    config = Config.from_template('gan')  # Dummy config
    pipeline = GraphPipeline(config)
    
    # Define workflow
    pipeline.add_step("load", load_data)
    pipeline.add_step("preprocess", preprocess, depends_on=["load"])
    pipeline.add_step("train", train_model, depends_on=["preprocess"])
    pipeline.add_step("evaluate", evaluate_model, depends_on=["preprocess", "train"])
    pipeline.add_step("visualize", visualize, depends_on=["train", "evaluate"])
    
    # Show execution plan
    print("\nExecution Plan:")
    for i, step in enumerate(pipeline.get_execution_plan(), 1):
        print(f"  {i}. {step}")
    
    # Execute
    print("\nExecuting Pipeline:")
    print("-" * 60)
    
    start = time.time()
    results = pipeline.run()
    duration = time.time() - start
    
    print("-" * 60)
    print(f"\n✓ Pipeline completed in {duration:.2f}s")
    
    # Show results
    print("\nResults:")
    for step_id, result in results.items():
        print(f"  {step_id}: {result}")
    
    print("\n" + "=" * 60)
    print("Example complete!")


if __name__ == '__main__':
    main()