"""Third-party integrations for FRAMEWORM"""

# Import with graceful fallback
try:
    from integrations.wandb import WandBIntegration, setup_wandb
except ImportError:
    WandBIntegration = None
    setup_wandb = None

try:
    from integrations.mlflow import MLflowIntegration, setup_mlflow
except ImportError:
    MLflowIntegration = None
    setup_mlflow = None

try:
    from integrations.storage import (
        S3Storage, GCSStorage, AzureStorage,
        setup_s3_storage, setup_gcs_storage, setup_azure_storage
    )
except ImportError:
    S3Storage = GCSStorage = AzureStorage = None
    setup_s3_storage = setup_gcs_storage = setup_azure_storage = None

try:
    from integrations.database import PostgresBackend
except ImportError:
    PostgresBackend = None

try:
    from integrations.notifications import SlackNotifier, EmailNotifier
except ImportError:
    SlackNotifier = EmailNotifier = None

__all__ = [
    'WandBIntegration', 'setup_wandb',
    'MLflowIntegration', 'setup_mlflow',
    'S3Storage', 'GCSStorage', 'AzureStorage',
    'setup_s3_storage', 'setup_gcs_storage', 'setup_azure_storage',
    'PostgresBackend',
    'SlackNotifier', 'EmailNotifier'
]