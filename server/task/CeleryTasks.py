import gc
import logging
import os
import socket
import time
import traceback

import GPUtil
from celery.exceptions import Reject
from celery.utils.log import get_task_logger

from mikazuki.app.models import APIResponse
from server.task.CeleryApp import celery_app
from server.task.CeleryTaskRequest import CeleryTaskRequest as TaskModel
from server.task.TaskExecutor import dispatch_run
from server.util.RedisClient import redis_client
import torch
# 获取服务器标识（例如主机名或IP地址）
#server_ip = socket.gethostname()
server_ip = os.getenv("SERVER_IP")

# 设置日志
logger = get_task_logger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def lock_gpu(gpu_memory_threshold=16, timeout=30,retry_interval=5,ex=100):
    t1 = int(time.time())
    while True:
        t2 = int(time.time())
        if t2 - t1 >= timeout:
            return None, None
        device_id, lock_name = do_lock_gpu(gpu_memory_threshold,ex)
        if device_id is not None:
            return device_id, lock_name
        time.sleep(retry_interval)

def do_lock_gpu(gpu_memory_threshold=16,ex=100):
    GPUs = GPUtil.getGPUs()
    allowd_gpu_ids=os.getenv("CUDA_VISIBLE_DEVICES")
    logger.info(f'allowd_gpu_ids:{allowd_gpu_ids}')
    for gpu in GPUs:
        if bool(allowd_gpu_ids)==False or str(gpu.id) in allowd_gpu_ids.split(","):
            logger.info(f'gpu:{gpu.id} ,free:{gpu.memoryFre}')
            if gpu.memoryFree > gpu_memory_threshold * 1024:
                lock_name = f"{server_ip}:GPU{gpu.id}"
                # 尝试获取锁
                if redis_client.set(
                        lock_name, "1", nx=True, ex=ex
                ):  # 如果成功获取锁，设置过期时间为100秒
                    return gpu.id, lock_name
    return None, None


@celery_app.task(
    bind=True,
    name="process_dataset",
    soft_time_limit=600,
    time_limit=2400,
    acks_late=True,
)
def process_dataset(
        self,
        task_id,
        taskType,
        taskConfig,

):
    task_id=self._get_request().id
    task = TaskModel(task_id=task_id, taskType=taskType, taskConfig=taskConfig)
    allowd_gpu_ids=os.getenv("CUDA_VISIBLE_DEVICES")
    device_id, lock_name = lock_gpu(gpu_memory_threshold=16)
    logger.info(f"{task_id} device_id, lock_name:{device_id, lock_name}")
    if device_id is not None:  # 如果设备可用且已模型初始化
        logger.info(f"process_dataset task {task_id}, {taskConfig}")
        try:
            customize_env = os.environ.copy()
            customize_env["ACCELERATE_DISABLE_RICH"] = "1"
            customize_env["PYTHONUNBUFFERED"] = "1"
            customize_env["CUDA_VISIBLE_DEVICES"] = f'{device_id}'
            customize_env["allowd_gpu_ids"] = allowd_gpu_ids
            logger.info(f"Using GPU(s) / 使用 GPU: {device_id}")
            ret = dispatch_run(task, customize_env)
            return ret
        except Exception as e:
            logger.exception(f"Task {task_id} failed due to an exception.")
            logger.exception(traceback.format_exc())
            raise self.retry(exc=e, countdown=10, max_retries=3)  # 10秒后重试
        finally:
            try:
              torch.cuda.empty_cache()
            except Exception as e:
                logger.exception(f"Task {task_id} empty_cache exception.")
            # 任务完成或失败后释放锁
            logger.info(f"task sucessed or failed for task {task_id}")
            try:
                redis_client.delete(lock_name)
            except Exception as e:
                logger.exception(f"Task {task_id} release lock error.")

    else:
        # 如果 GPU 不可用或内存不足，将任务重新放回队列
        raise Reject("GPU resource unavailable", requeue=True)


@celery_app.task(
    bind=True,
    name="process_loratrain",
    soft_time_limit=1800,
    time_limit=2400,
    acks_late=True,
)
def process_loratrain(
        self,
        task_id,
        taskType,
        taskConfig,

):
    task_id = self._get_request().id
    task = TaskModel(task_id=task_id, taskType=taskType, taskConfig=taskConfig)
    device_id, lock_name = lock_gpu(gpu_memory_threshold=18,ex=1800)
    logger.info(f"{task_id} device_id, lock_name:{device_id, lock_name}")
    if device_id is not None:  # 如果设备可用且已模型初始化
        logger.info(f"process_loratrain task {task_id}, {taskConfig}")
        try:
            customize_env = os.environ.copy()
            customize_env["ACCELERATE_DISABLE_RICH"] = "1"
            customize_env["PYTHONUNBUFFERED"] = "1"
            customize_env["CUDA_VISIBLE_DEVICES"] = f'{device_id}'
            logger.info(f"Using GPU(s) / 使用 GPU: {device_id}")
            ret = dispatch_run(task, customize_env)
            return ret
        except Exception as e:
            logger.exception(f"Task {task_id} failed due to an exception.")
            logger.exception(traceback.format_exc())
            raise self.retry(exc=e, countdown=30, max_retries=3)  # 30秒后重试
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            # 任务完成或失败后释放锁
            logger.info(f"task sucessed or failed for task {task_id}")
            redis_client.delete(lock_name)
    else:
        # 如果 GPU 不可用或内存不足，将任务重新放回队列
        raise Reject("GPU resource unavailable", requeue=True)
