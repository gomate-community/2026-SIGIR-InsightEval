import os
import asyncio
from typing import List, Optional, Dict, Any, BinaryIO
from datetime import datetime
from loguru import logger

from minio import Minio
from minio.error import S3Error


class MinioEngine:
    """MinIO 对象存储服务，用于PDF文件的上传、下载和管理"""

    def __init__(
        self,
        endpoint: str = "127.0.0.1:9000",
        access_key: str = "minioadmin",
        secret_key: str = "minioadmin",
        secure: bool = False,
        default_bucket: str = "pdf-files"
    ):
        """
        初始化MinIO服务

        Args:
            endpoint: MinIO服务器地址
            access_key: 访问密钥
            secret_key: 秘密密钥
            secure: 是否使用HTTPS
            default_bucket: 默认存储桶名称
        """
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.default_bucket = default_bucket

        # 初始化MinIO客户端
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )

        # 延迟初始化存储桶，避免在__init__中创建async任务
        self._bucket_initialized = False

    async def ensure_bucket_exists(self, bucket_name: str) -> None:
        """
        确保存储桶存在，如果不存在则创建

        Args:
            bucket_name: 存储桶名称
        """
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"创建MinIO存储桶: {bucket_name}")
        except S3Error as e:
            logger.error(f"创建MinIO存储桶失败: {bucket_name}, 错误: {e}")
            raise

    async def _ensure_default_bucket(self) -> None:
        """
        确保默认存储桶存在（延迟初始化）
        """
        if not self._bucket_initialized:
            await self.ensure_bucket_exists(self.default_bucket)
            self._bucket_initialized = True

    async def upload_file(
        self,
        file_path: str,
        object_name: str,
        content_type: str = "application/octet-stream",
        bucket_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        上传文件到MinIO

        Args:
            file_path: 本地文件路径
            object_name: 对象名称
            content_type: 内容类型
            bucket_name: 存储桶名称，默认使用配置的存储桶

        Returns:
            上传结果字典
        """
        bucket = bucket_name or self.default_bucket

        # 确保存储桶存在
        await self._ensure_default_bucket()

        try:
            logger.error(f"创建存储桶失败 {bucket_name}: {str(e)}")
        except Exception as e:
            logger.error(f"检查存储桶存在性失败 {bucket_name}: {str(e)}")

    async def upload_file(
        self,
        file_path: str,
        object_name: Optional[str] = None,
        bucket_name: Optional[str] = None,
        content_type: str = "application/pdf"
    ) -> Dict[str, Any]:
        """
        上传文件到MinIO

        Args:
            file_path: 本地文件路径
            object_name: 对象名称（在MinIO中的文件名），如果为None则使用原文件名
            bucket_name: 存储桶名称，如果为None则使用默认存储桶
            content_type: 文件内容类型

        Returns:
            上传结果信息
        """
        bucket = bucket_name or self.default_bucket
        obj_name = object_name or os.path.basename(file_path)

        try:
            # 确保存储桶存在
            await self.ensure_bucket_exists(bucket)

            # 上传文件
            result = self.client.fput_object(
                bucket_name=bucket,
                object_name=obj_name,
                file_path=file_path,
                content_type=content_type
            )

            logger.info(f"成功上传文件到MinIO: {bucket}/{obj_name}")

            return {
                "success": True,
                "bucket": bucket,
                "object_name": obj_name,
                "file_path": file_path,
                "uploaded_at": datetime.now().isoformat(),
                "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }

        except S3Error as e:
            logger.error(f"上传文件到MinIO失败: {str(e)}")
            return {
                "success": False,
                "bucket": bucket,
                "object_name": obj_name,
                "error": str(e),
                "uploaded_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"上传文件异常: {str(e)}")
            return {
                "success": False,
                "bucket": bucket,
                "object_name": obj_name,
                "error": str(e),
                "uploaded_at": datetime.now().isoformat()
            }

    async def upload_bytes(
        self,
        data: bytes,
        object_name: str,
        bucket_name: Optional[str] = None,
        content_type: str = "application/pdf"
    ) -> Dict[str, Any]:
        """
        上传字节数据到MinIO

        Args:
            data: 字节数据
            object_name: 对象名称
            bucket_name: 存储桶名称，如果为None则使用默认存储桶
            content_type: 文件内容类型

        Returns:
            上传结果信息
        """
        from io import BytesIO

        bucket = bucket_name or self.default_bucket

        try:
            # 确保存储桶存在
            await self.ensure_bucket_exists(bucket)

            # 上传字节数据
            data_stream = BytesIO(data)
            result = self.client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=data_stream,
                length=len(data),
                content_type=content_type
            )

            logger.info(f"成功上传字节数据到MinIO: {bucket}/{object_name}")

            return {
                "success": True,
                "bucket": bucket,
                "object_name": object_name,
                "uploaded_at": datetime.now().isoformat(),
                "size": len(data)
            }

        except S3Error as e:
            logger.error(f"上传字节数据到MinIO失败: {str(e)}")
            return {
                "success": False,
                "bucket": bucket,
                "object_name": object_name,
                "error": str(e),
                "uploaded_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"上传字节数据异常: {str(e)}")
            return {
                "success": False,
                "bucket": bucket,
                "object_name": object_name,
                "error": str(e),
                "uploaded_at": datetime.now().isoformat()
            }

    async def download_file(
        self,
        object_name: str,
        file_path: str,
        bucket_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        从MinIO下载文件

        Args:
            object_name: 对象名称
            file_path: 本地保存路径
            bucket_name: 存储桶名称，如果为None则使用默认存储桶

        Returns:
            下载结果信息
        """
        bucket = bucket_name or self.default_bucket

        try:
            # 下载文件
            self.client.fget_object(
                bucket_name=bucket,
                object_name=object_name,
                file_path=file_path
            )

            logger.info(f"成功从MinIO下载文件: {bucket}/{object_name} -> {file_path}")

            return {
                "success": True,
                "bucket": bucket,
                "object_name": object_name,
                "file_path": file_path,
                "downloaded_at": datetime.now().isoformat(),
                "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }

        except S3Error as e:
            logger.error(f"从MinIO下载文件失败: {str(e)}")
            return {
                "success": False,
                "bucket": bucket,
                "object_name": object_name,
                "file_path": file_path,
                "error": str(e),
                "downloaded_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"下载文件异常: {str(e)}")
            return {
                "success": False,
                "bucket": bucket,
                "object_name": object_name,
                "file_path": file_path,
                "error": str(e),
                "downloaded_at": datetime.now().isoformat()
            }

    async def get_object_bytes(
        self,
        object_name: str,
        bucket_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取对象的字节数据

        Args:
            object_name: 对象名称
            bucket_name: 存储桶名称，如果为None则使用默认存储桶

        Returns:
            包含字节数据的字典
        """
        from io import BytesIO

        bucket = bucket_name or self.default_bucket

        try:
            # 获取对象数据
            response = self.client.get_object(bucket_name=bucket, object_name=object_name)
            data = response.read()

            logger.info(f"成功从MinIO获取字节数据: {bucket}/{object_name}")

            return {
                "success": True,
                "bucket": bucket,
                "object_name": object_name,
                "data": data,
                "size": len(data),
                "retrieved_at": datetime.now().isoformat()
            }

        except S3Error as e:
            logger.error(f"从MinIO获取字节数据失败: {str(e)}")
            return {
                "success": False,
                "bucket": bucket,
                "object_name": object_name,
                "error": str(e),
                "retrieved_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取字节数据异常: {str(e)}")
            return {
                "success": False,
                "bucket": bucket,
                "object_name": object_name,
                "error": str(e),
                "retrieved_at": datetime.now().isoformat()
            }

    async def list_objects(
        self,
        bucket_name: Optional[str] = None,
        prefix: str = ""
    ) -> List[Dict[str, Any]]:
        """
        列出存储桶中的对象

        Args:
            bucket_name: 存储桶名称，如果为None则使用默认存储桶
            prefix: 对象前缀过滤

        Returns:
            对象列表
        """
        bucket = bucket_name or self.default_bucket
        objects = []

        try:
            # 如果使用默认存储桶，确保其存在
            if bucket_name is None:
                await self._ensure_default_bucket()

            # 获取对象列表
            obj_iter = self.client.list_objects(bucket_name=bucket, prefix=prefix)

            for obj in obj_iter:
                objects.append({
                    "object_name": obj.object_name,
                    "size": obj.size,
                    "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                    "etag": obj.etag,
                    "bucket": bucket
                })

            logger.info(f"成功列出MinIO对象: {bucket} (共{len(objects)}个对象)")

            return objects

        except S3Error as e:
            logger.error(f"列出MinIO对象失败: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"列出对象异常: {str(e)}")
            return []

    async def remove_object(
        self,
        object_name: str,
        bucket_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        删除MinIO中的对象

        Args:
            object_name: 对象名称
            bucket_name: 存储桶名称，如果为None则使用默认存储桶

        Returns:
            删除结果信息
        """
        bucket = bucket_name or self.default_bucket

        try:
            # 如果使用默认存储桶，确保其存在
            if bucket_name is None:
                await self._ensure_default_bucket()

            # 删除对象
            self.client.remove_object(bucket_name=bucket, object_name=object_name)

            logger.info(f"成功删除MinIO对象: {bucket}/{object_name}")

            return {
                "success": True,
                "bucket": bucket,
                "object_name": object_name,
                "deleted_at": datetime.now().isoformat()
            }

        except S3Error as e:
            logger.error(f"删除MinIO对象失败: {str(e)}")
            return {
                "success": False,
                "bucket": bucket,
                "object_name": object_name,
                "error": str(e),
                "deleted_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"删除对象异常: {str(e)}")
            return {
                "success": False,
                "bucket": bucket,
                "object_name": object_name,
                "error": str(e),
                "deleted_at": datetime.now().isoformat()
            }

    async def object_exists(
        self,
        object_name: str,
        bucket_name: Optional[str] = None
    ) -> bool:
        """
        检查对象是否存在

        Args:
            object_name: 对象名称
            bucket_name: 存储桶名称，如果为None则使用默认存储桶

        Returns:
            对象是否存在
        """
        bucket = bucket_name or self.default_bucket

        try:
            # 尝试获取对象信息来检查是否存在
            self.client.stat_object(bucket_name=bucket, object_name=object_name)
            return True
        except S3Error:
            return False
        except Exception as e:
            logger.error(f"检查对象存在性异常: {str(e)}")
            return False

    def list_buckets(self) -> List[Dict[str, Any]]:
        """
        列出所有存储桶

        Returns:
            存储桶列表
        """
        try:
            buckets = self.client.list_buckets()
            bucket_list = []

            for bucket in buckets:
                bucket_list.append({
                    "name": bucket.name,
                    "creation_date": bucket.creation_date.isoformat() if bucket.creation_date else None
                })

            logger.info(f"成功列出MinIO存储桶: 共{len(bucket_list)}个存储桶")
            return bucket_list

        except S3Error as e:
            logger.error(f"列出存储桶失败: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"列出存储桶异常: {str(e)}")
            return []

    async def health_check(self) -> bool:
        """
        健康检查

        Returns:
            服务是否正常
        """
        try:
            # 尝试列出存储桶来检查连接
            self.client.list_buckets()
            return True
        except Exception as e:
            logger.error(f"MinIO服务健康检查失败: {str(e)}")
            return False
