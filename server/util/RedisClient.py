import os

import redis
from celery.utils.log import get_task_logger


class RedisClient:
    def __init__(self):
        self.logger = get_task_logger(__name__)
        self.redis_host = os.environ.get("LORA_REDIS_HOST")
        self.redis_port = os.environ.get("LORA_REDIS_PORT")
        self.redis_auth = os.environ.get("LORA_REDIS_AUTH")
        self.redis_db = os.environ.get("LORA_REDIS_DB")
        self.redis_client = redis.Redis(host=self.redis_host, port=self.redis_port, password=self.redis_auth,
                                        db=self.redis_db)

    def instance(self):
        return self.redis_client


RedisClientClz = RedisClient()
redis_client = RedisClientClz.instance()
