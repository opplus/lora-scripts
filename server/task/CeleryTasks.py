import gc
import logging
import os
import time
import traceback
from typing import Tuple, List, Any
import uuid
import GPUtil
import torch
from celery.exceptions import Reject
from celery.utils.log import get_task_logger

from server.task.CeleryApp import celery_app
from server.task.CeleryTaskRequest import CeleryTaskRequest as TaskModel
from server.task.TaskExecutor import dispatch_run
from server.util.RedisClient import redis_client, RedisClientClz
from server.util.process_util import interrupt_current_processing

# 获取服务器标识（例如主机名或IP地址）
# server_ip = socket.gethostname()
server_ip = os.getenv("SERVER_IP")
# 新增唯一客户端标识[5,7](@ref)
CLIENT_ID = f"{server_ip}-{uuid.uuid4().hex[:8]}"
# 设置日志
logger = get_task_logger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
def max_gpu_num():
    maxGpuNum = os.getenv("LORA_MAX_GPU_NUM")
    if maxGpuNum is None:
        return 1
    return maxGpuNum

def _is_real_gpuid():
    allowd_gpu_ids = os.getenv("LORA_ALLOWD_GPU_IDS")
    if allowd_gpu_ids is None or len(str.strip(allowd_gpu_ids)) == 0:
        return False
    return True


def _get_allow_gpu_ids():
    allowd_gpu_ids = os.getenv("LORA_ALLOWD_GPU_IDS")
    if allowd_gpu_ids is None or len(str.strip(allowd_gpu_ids)) == 0:
        allowd_gpu_ids = os.getenv("CUDA_VISIBLE_DEVICES")
    logger.info(f'allowd_gpu_ids:{allowd_gpu_ids}')
    return allowd_gpu_ids



def lock_gpu(support_max_gpu_num=1, gpu_memory_threshold=16, timeout=30,
             retry_interval=5, ex=100) -> Tuple[List[int], List[str]]:
    """支持同时锁定多个GPU的版本"""
    acquired_devices = []
    acquired_locks = []
    t1 = time.time()

    while time.time() - t1 < timeout:
        # 新增清空临时存储[2](@ref)
        tmp_devices = []
        tmp_locks = []

        # 获取可用GPU列表
        GPUs = GPUtil.getGPUs()
        allowd_gpu_ids = _get_allow_gpu_ids()

        # 新增重试时释放残留锁的机制[2](@ref)
        try:
            for gpu in GPUs:
                if len(tmp_devices) >= support_max_gpu_num:
                    break

                if _is_gpu_available(gpu, allowd_gpu_ids, gpu_memory_threshold):
                    device_id, lock_name = _try_acquire_lock(gpu, ex)
                    if device_id is not None:
                        tmp_devices.append(device_id)
                        tmp_locks.append(lock_name)

            # 成功获取足够数量锁[1](@ref)
            if len(tmp_devices) >= support_max_gpu_num:
                acquired_devices = tmp_devices
                acquired_locks = tmp_locks
                return acquired_devices, acquired_locks

        finally:
            # 部分获取失败时释放已获得的锁[2](@ref)
            if len(tmp_devices) > 0:
                release_gpu(tmp_locks)

        time.sleep(retry_interval)

    return [], []


def _try_acquire_lock(gpu, ex) -> tuple[Any, str] | tuple[None, None]:
    """带唯一标识的锁获取"""
    lock_name = f"{server_ip}:GPU{gpu.id}"
    # 改用唯一值作为value[5,7](@ref)
    unique_token = f"{CLIENT_ID}"
    if redis_client.set(
            lock_name,
            unique_token,
            nx=True,
            ex=ex
    ):
        return gpu.id, lock_name
    return None, None


def release_gpu(lock_names: List[str]):
    """批量释放锁的原子操作"""
    if not lock_names:
        return

    # 使用Lua脚本保证原子性[5,7](@ref)
    lua_script = """
    for i, key in ipairs(KEYS) do
        if redis.call("get", key) == ARGV[1] then
            redis.call("del", key)
        end
    end
    return 1
    """
    try:
        redis_client.eval(
            lua_script,
            len(lock_names),
            *lock_names,
            CLIENT_ID
        )
    except Exception as e:
        logger.error(f"释放锁失败: {str(e)}")


# 新增辅助函数
def _is_gpu_available(gpu, allowd_gpu_ids, threshold) -> bool:
    """判断GPU是否满足条件"""
    memory_ok = gpu.memoryFree > threshold * 1024
    id_ok = not allowd_gpu_ids or str(gpu.id) in allowd_gpu_ids.split(",")
    return memory_ok and id_ok


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
    task_id = self._get_request().id
    task = TaskModel(task_id=task_id, taskType=taskType, taskConfig=taskConfig)
    allowd_gpu_ids = _get_allow_gpu_ids()
    device_ids, lock_names = lock_gpu(support_max_gpu_num=1, gpu_memory_threshold=16, ex=180)
    logger.info(f"{task_id} device_id, lock_name:{device_ids, lock_names}")
    if device_ids is not None and len(device_ids) > 0:
        device_ids_str=",".join([str(num) for num in device_ids])
        logger.info(f"process_dataset task {task_id}, {taskConfig}")
        try:
            customize_env = os.environ.copy()
            customize_env["ACCELERATE_DISABLE_RICH"] = "1"
            customize_env["PYTHONUNBUFFERED"] = "1"
            customize_env["is_real_gpuid"] = "1" if _is_real_gpuid() else "0"
            # device_ids转成逗号分割
            customize_env["CUDA_VISIBLE_DEVICES"] = f'{device_ids_str}'
            customize_env["allowd_gpu_ids"] = allowd_gpu_ids
            logger.info(f"Using GPU(s) / 使用 GPU: {device_ids_str}")
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
                release_gpu(lock_names)
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
    interrupt_current_processing(value=False)
    task_id = self._get_request().id
    task = TaskModel(task_id=task_id, taskType=taskType, taskConfig=taskConfig)
    allowd_gpu_ids = _get_allow_gpu_ids()
    support_max_gpu_num = max_gpu_num()
    device_ids, lock_names = lock_gpu(support_max_gpu_num, gpu_memory_threshold=18, ex=1800)
    logger.info(f"{task_id} device_id, lock_name:{device_ids, lock_names}")
    if device_ids is not None and len(device_ids)>0:
        device_ids_str = ",".join([str(num) for num in device_ids])
        logger.info(f"process_loratrain task {task_id}, {taskConfig}")
        try:
            customize_env = os.environ.copy()
            customize_env["ACCELERATE_DISABLE_RICH"] = "1"
            customize_env["PYTHONUNBUFFERED"] = "1"
            customize_env["CUDA_VISIBLE_DEVICES"] = f'{device_ids_str}'
            customize_env["allowd_gpu_ids"] = allowd_gpu_ids
            customize_env["is_real_gpuid"] = "1" if _is_real_gpuid() else "0"
            logger.info(f"Using GPU(s) / 使用 GPU: {device_ids_str}")
            ret = dispatch_run(task, customize_env)
            return ret
        except Exception as e:
            logger.exception(f"Task {task_id} failed due to an exception.")
            logger.exception(traceback.format_exc())
            raise self.retry(exc=e, countdown=30, max_retries=3)  # 30秒后重试
        finally:
            # 任务完成或失败后释放锁
            logger.info(f"task sucessed or failed for task {task_id}")
            release_gpu(lock_names)
            interrupt_current_processing(value=True)
            gc.collect()
            torch.cuda.empty_cache()
    else:
        # 如果 GPU 不可用或内存不足，将任务重新放回队列
        raise Reject("GPU resource unavailable", requeue=True)
