"""
Cloud storage integrations (S3, Google Cloud Storage, Azure Blob).

Automatically upload checkpoints and artifacts to cloud storage.

Example:
    >>> from frameworm.integrations import S3Storage
    >>> 
    >>> storage = S3Storage(
    ...     bucket='my-ml-bucket',
    ...     prefix='experiments/vae-mnist'
    ... )
    >>> trainer.add_callback(storage)
"""

from typing import Optional
from pathlib import Path
from training.callbacks import Callback


class S3Storage(Callback):
    """
    AWS S3 storage integration.
    
    Automatically uploads checkpoints and artifacts to S3.
    
    Args:
        bucket: S3 bucket name
        prefix: Key prefix (folder path)
        region: AWS region (default: us-east-1)
        access_key: AWS access key (or use env AWS_ACCESS_KEY_ID)
        secret_key: AWS secret key (or use env AWS_SECRET_ACCESS_KEY)
        
    Example:
        >>> s3 = S3Storage(
        ...     bucket='my-models',
        ...     prefix='frameworm/vae-mnist'
        ... )
        >>> trainer.add_callback(s3)
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str = '',
        region: str = 'us-east-1',
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None
    ):
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 not installed. Install: pip install boto3")
        
        self.bucket = bucket
        self.prefix = prefix
        
        # Initialize S3 client
        if access_key and secret_key:
            self.s3 = boto3.client(
                's3',
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
        else:
            # Use environment variables or IAM role
            self.s3 = boto3.client('s3', region_name=region)
    
    def on_checkpoint_save(self, trainer, path):
        """Upload checkpoint to S3"""
        key = f"{self.prefix}/{Path(path).name}".lstrip('/')
        
        try:
            self.s3.upload_file(path, self.bucket, key)
            s3_url = f"s3://{self.bucket}/{key}"
            print(f"✓ Uploaded to S3: {s3_url}")
        except Exception as e:
            print(f"⚠️  S3 upload failed: {e}")
    
    def download_checkpoint(self, key: str, local_path: str):
        """Download checkpoint from S3"""
        full_key = f"{self.prefix}/{key}".lstrip('/')
        self.s3.download_file(self.bucket, full_key, local_path)
        print(f"✓ Downloaded from S3: {local_path}")


class GCSStorage(Callback):
    """
    Google Cloud Storage integration.
    
    Args:
        bucket: GCS bucket name
        prefix: Object prefix (folder path)
        credentials_path: Path to service account JSON (optional)
        
    Example:
        >>> gcs = GCSStorage(
        ...     bucket='my-ml-models',
        ...     prefix='frameworm/experiments'
        ... )
        >>> trainer.add_callback(gcs)
    """
    
    def __init__(
        self,
        bucket: str,
        prefix: str = '',
        credentials_path: Optional[str] = None
    ):
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "google-cloud-storage not installed. "
                "Install: pip install google-cloud-storage"
            )
        
        self.bucket_name = bucket
        self.prefix = prefix
        
        # Initialize GCS client
        if credentials_path:
            self.client = storage.Client.from_service_account_json(credentials_path)
        else:
            # Use default credentials
            self.client = storage.Client()
        
        self.bucket = self.client.bucket(bucket)
    
    def on_checkpoint_save(self, trainer, path):
        """Upload checkpoint to GCS"""
        blob_name = f"{self.prefix}/{Path(path).name}".lstrip('/')
        blob = self.bucket.blob(blob_name)
        
        try:
            blob.upload_from_filename(path)
            gcs_url = f"gs://{self.bucket_name}/{blob_name}"
            print(f"✓ Uploaded to GCS: {gcs_url}")
        except Exception as e:
            print(f"⚠️  GCS upload failed: {e}")
    
    def download_checkpoint(self, blob_name: str, local_path: str):
        """Download checkpoint from GCS"""
        full_name = f"{self.prefix}/{blob_name}".lstrip('/')
        blob = self.bucket.blob(full_name)
        blob.download_to_filename(local_path)
        print(f"✓ Downloaded from GCS: {local_path}")


class AzureStorage(Callback):
    """
    Azure Blob Storage integration.
    
    Args:
        account_name: Azure storage account name
        container: Container name
        prefix: Blob prefix (folder path)
        account_key: Account key (or use env AZURE_STORAGE_KEY)
        
    Example:
        >>> azure = AzureStorage(
        ...     account_name='myaccount',
        ...     container='ml-models'
        ... )
        >>> trainer.add_callback(azure)
    """
    
    def __init__(
        self,
        account_name: str,
        container: str,
        prefix: str = '',
        account_key: Optional[str] = None
    ):
        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError:
            raise ImportError(
                "azure-storage-blob not installed. "
                "Install: pip install azure-storage-blob"
            )
        
        self.container_name = container
        self.prefix = prefix
        
        # Initialize client
        if account_key:
            connection_string = (
                f"DefaultEndpointsProtocol=https;"
                f"AccountName={account_name};"
                f"AccountKey={account_key};"
                f"EndpointSuffix=core.windows.net"
            )
            self.client = BlobServiceClient.from_connection_string(connection_string)
        else:
            # Use environment variable or managed identity
            import os
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            self.client = BlobServiceClient.from_connection_string(connection_string)
        
        self.container_client = self.client.get_container_client(container)
    
    def on_checkpoint_save(self, trainer, path):
        """Upload checkpoint to Azure"""
        blob_name = f"{self.prefix}/{Path(path).name}".lstrip('/')
        blob_client = self.container_client.get_blob_client(blob_name)
        
        try:
            with open(path, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            
            azure_url = f"https://{self.client.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}"
            print(f"✓ Uploaded to Azure: {azure_url}")
        except Exception as e:
            print(f"⚠️  Azure upload failed: {e}")


# Convenience functions
def setup_s3_storage(bucket: str, prefix: str = '', **kwargs) -> S3Storage:
    """Quick S3 setup"""
    return S3Storage(bucket=bucket, prefix=prefix, **kwargs)


def setup_gcs_storage(bucket: str, prefix: str = '', **kwargs) -> GCSStorage:
    """Quick GCS setup"""
    return GCSStorage(bucket=bucket, prefix=prefix, **kwargs)


def setup_azure_storage(account_name: str, container: str, prefix: str = '', **kwargs) -> AzureStorage:
    """Quick Azure setup"""
    return AzureStorage(account_name=account_name, container=container, prefix=prefix, **kwargs)