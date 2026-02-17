import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiment.experiment import Experiment
from training.trainer import Trainer


def test_trainer_experiment_integration(tmp_path):
    """
    Integration test for Trainer + Experiment logging.
    Ensures:
    - training runs
    - experiment context works
    - metrics are logged
    """

    # Dummy model
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

        def compute_loss(self, x, y):
            return {"loss": nn.MSELoss()(self.forward(x), y)}

    # Dummy dataset
    X = torch.randn(50, 10)
    y = torch.randn(50, 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=10)

    # Temporary experiment directory
    root_dir = tmp_path / "test_experiments"

    # Create experiment
    exp = Experiment(name="trainer-test", config={"lr": 0.001}, root_dir=str(root_dir))

    # Setup trainer
    model = Model()
    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(model, optimizer, device="cpu")
    trainer.set_experiment(exp)

    # Run training inside experiment context
    with exp:
        trainer.train(loader, epochs=2)

    # Assertions

    # Experiment completed successfully
    assert exp.status in ["completed", "finished", "success"]

    # Metrics were logged
    metrics = exp.get_metrics()
    assert len(metrics) > 0

    # Training state updated
    assert trainer.state.current_epoch == 2
