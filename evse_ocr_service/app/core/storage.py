import os
from abc import ABC, abstractmethod
from typing import Optional
from app.config import settings
from app.core.logging import logger

class StorageService(ABC):
    @abstractmethod
    async def upload_image(self, file_bytes: bytes, file_id: str, content_type: str) -> str:
        pass

    @abstractmethod
    async def download_image(self, uri: str) -> bytes:
        pass

class LocalStorageService(StorageService):
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or settings.LOCAL_STORAGE_DIR
        os.makedirs(self.base_dir, exist_ok=True)

    async def upload_image(self, file_bytes: bytes, file_id: str, content_type: str) -> str:
        ext = ".jpg" if "jpeg" in content_type else (".png" if "png" in content_type else ".bin")
        filename = f"{file_id}{ext}"
        filepath = os.path.join(self.base_dir, filename)
        with open(filepath, "wb") as f:
            f.write(file_bytes)
        logger.info(f"Persisted file locally: {filepath}")
        return f"file://{os.path.abspath(filepath)}"

    async def download_image(self, uri: str) -> bytes:
        filepath = uri.replace("file://", "")
        with open(filepath, "rb") as f:
            return f.read()

class S3StorageService(StorageService):
    def __init__(self):
        try:
            import boto3
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=settings.S3_ENDPOINT_URL,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION,
            )
            self.bucket_name = settings.S3_BUCKET_NAME
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.s3_client = None

    async def upload_image(self, file_bytes: bytes, file_id: str, content_type: str) -> str:
        if not self.s3_client:
            return await LocalStorageService().upload_image(file_bytes, file_id, content_type)
            
        ext = ".jpg" if "jpeg" in content_type else (".png" if "png" in content_type else ".bin")
        object_key = f"scans/{file_id}{ext}"
        
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=object_key,
            Body=file_bytes,
            ContentType=content_type
        )
        s3_uri = f"s3://{self.bucket_name}/{object_key}"
        logger.info(f"Uploaded image to S3: {s3_uri}")
        return s3_uri

    async def download_image(self, uri: str) -> bytes:
        if not self.s3_client or not uri.startswith("s3://"):
            return await LocalStorageService().download_image(uri)
            
        parts = uri.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()

def get_storage_service() -> StorageService:
    if settings.STORAGE_TYPE.lower() == "s3":
        return S3StorageService()
    return LocalStorageService()
