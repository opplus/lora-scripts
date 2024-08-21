import os

from celery import Celery
from kombu import Queue


class CeleryApp:
    def __init__(self):
        self.celery_app = Celery(
            "worker",
            backend=os.environ["LORA_CELERY_BACKEND"],
            broker=os.environ["LORA_CELERY_BROKER"],
            broker_connection_retry_on_startup=False,
        )
        # 任务执行完成后再确认,避免任务丢失
        self.celery_app.conf.task_acks_late = True
        # worker 失联时将任务标记为失败并重新入队,提高可靠性
        self.celery_app.conf.task_reject_on_worker_lost = True
        # 设置时区
        self.celery_app.conf.timezone = 'Asia/Shanghai'
        self.celery_app.conf.enable_utc = False

        # 定义队列
        self.celery_app.conf.task_queues = (
            Queue('process_loratrain'),
            Queue('process_dataset'),
        )

        # 配置任务路由
        self.celery_app.conf.task_routes = {
            'process_loratrain': {'queue': 'process_loratrain'},
            'process_dataset': {'queue': 'process_dataset'},
        }

    def get_celery_app(self):
        return self.celery_app


celery_app = CeleryApp().get_celery_app()
