import os
import oss2


class OSSManager:
    """创建oss桶"""

    def __init__(self):
        # 从环境变量中获取OSS配置
        self.access_key_id = os.environ.get("LORA_OSS_ACCESS_KEY_ID")
        self.access_key_secret = os.environ.get("LORA_OSS_ACCESS_KEY_SECRET")
        self.endpoint = os.environ.get("LORA_OSS_ENDPOINT")
        self.bucket_name = os.environ.get("LORA_OSS_BUCKET_NAME")

        # OSS认证
        self.auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)

    def upload_to_oss(self, oss_image_path, local_image_path):
        """上传文件到OSS"""
        self.bucket.put_object_from_file(oss_image_path, local_image_path)

    def download_from_oss(self, oss_image_path, local_image_path):
        """从OSS下载文件"""
        self.bucket.get_object_to_file(oss_image_path, local_image_path)


oss = OSSManager()
