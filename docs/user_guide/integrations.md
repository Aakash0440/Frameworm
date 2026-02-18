# Third-Party Integrations

FRAMEWORM integrates with popular ML tools and services.

---

## Experiment Tracking

### Weights & Biases
```python
from frameworm.integrations import WandBIntegration

wandb_logger = WandBIntegration(
    project='my-project',
    entity='my-team',
    config=config.to_dict(),
    tags=['vae', 'experiment-1']
)

trainer.add_callback(wandb_logger)
trainer.train(...)  # Auto-logs to W&B
```

### MLflow
```python
from frameworm.integrations import MLflowIntegration

mlflow_logger = MLflowIntegration(
    experiment_name='vae-training',
    tracking_uri='http://mlflow-server:5000'
)

trainer.add_callback(mlflow_logger)
```

---

## Cloud Storage

### AWS S3
```python
from frameworm.integrations import S3Storage

s3 = S3Storage(
    bucket='my-ml-models',
    prefix='experiments/vae',
    region='us-east-1'
)

trainer.add_callback(s3)  # Auto-uploads checkpoints
```

### Google Cloud Storage
```python
from frameworm.integrations import GCSStorage

gcs = GCSStorage(
    bucket='my-ml-models',
    prefix='experiments',
    credentials_path='service-account.json'
)

trainer.add_callback(gcs)
```

### Azure Blob Storage
```python
from frameworm.integrations import AzureStorage

azure = AzureStorage(
    account_name='myaccount',
    container='ml-models'
)

trainer.add_callback(azure)
```

---

## Notifications

### Slack
```python
from frameworm.integrations import SlackNotifier

slack = SlackNotifier(
    webhook_url='https://hooks.slack.com/...',
    notify_on_epoch=10  # Every 10 epochs
)

trainer.add_callback(slack)
```

### Email
```python
from frameworm.integrations import EmailNotifier

email = EmailNotifier(
    smtp_server='smtp.gmail.com',
    sender_email='your@gmail.com',
    sender_password='app_password',
    recipient_email='notify@example.com'
)

trainer.add_callback(email)
```

---

## Complete Example
```python
from frameworm import Trainer, Config, get_model
from frameworm.integrations import (
    WandBIntegration,
    S3Storage,
    SlackNotifier
)

# Setup integrations
wandb = WandBIntegration(project='my-project')
s3 = S3Storage(bucket='my-models', prefix='exp1')
slack = SlackNotifier(webhook_url='...')

# Train with all integrations
trainer = Trainer(model, optimizer)
trainer.add_callback(wandb)
trainer.add_callback(s3)
trainer.add_callback(slack)

trainer.train(train_loader, val_loader, epochs=100)
```

Result:
- ✅ Metrics logged to W&B
- ✅ Checkpoints uploaded to S3
- ✅ Slack notifications on completion