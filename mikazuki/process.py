
import asyncio
import os
import sys
from typing import Optional

from mikazuki.app.models import APIResponse
from mikazuki.log import log
from mikazuki.tasks import tm
import GPUtil
server_ip = os.getenv("SERVER_IP")

def lock_gpu(gpu_memory_threshold=20):
    GPUs = GPUtil.getGPUs()
    allowd_gpu_ids=os.getenv("CUDA_VISIBLE_DEVICES")
    log.info(f'allowd_gpu_ids:{allowd_gpu_ids}')
    for gpu in GPUs:
        log.info(f'gpu:{gpu.id}')
    for gpu in GPUs:
        if bool(allowd_gpu_ids)==False or str(gpu.id) in allowd_gpu_ids.split(","):
            if gpu.memoryFree > gpu_memory_threshold * 1024:
                lock_name = f"{server_ip}:GPU{gpu.id}"
                from server.util.RedisClient import redis_client
                # 尝试获取锁
                if redis_client.set(
                        lock_name, "1", nx=True, ex=100
                ):  # 如果成功获取锁，设置过期时间为100秒
                    return gpu.id, lock_name
    return None, None


def run_train(toml_path: str,
              trainer_file: str = "./sd-scripts/train_network.py",
              gpu_ids: Optional[list] = None,
              cpu_threads: Optional[int] = 2):
    device_id, lock_name = lock_gpu()
    log.info(f"device_id, lock_name:{device_id, lock_name}")
    if device_id is not None:  # 如果设备可用
        try:
            return run_train_0(toml_path=toml_path,trainer_file=trainer_file,gpu_ids=[str(device_id)])
        finally:
            from server.util.RedisClient import redis_client
            redis_client.delete(lock_name)
    else:
        log.error(f"An error occurred when training / 训练无可用资源，请稍后再试")
        return APIResponse(status="fail", message=f"训练无可用资源，请稍后再试")


def run_train_0(toml_path: str,
              trainer_file: str = "./sd-scripts/train_network.py",
              gpu_ids: Optional[list] = None,
              cpu_threads: Optional[int] = 2):
    log.info(f"Training started with config file / 训练开始，使用配置文件: {toml_path}")
    args = [
        sys.executable, "-m", "accelerate.commands.launch",  # use -m to avoid python script executable error
        "--num_cpu_threads_per_process", str(cpu_threads),  # cpu threads
        "--quiet",  # silence accelerate error message
        trainer_file,
        "--config_file", toml_path,
    ]

    customize_env = os.environ.copy()
    customize_env["ACCELERATE_DISABLE_RICH"] = "1"
    customize_env["PYTHONUNBUFFERED"] = "1"

    if gpu_ids:
        customize_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        log.info(f"Using GPU(s) / 使用 GPU: {gpu_ids}")

        if len(gpu_ids) > 1:
            args[3:3] = ["--multi_gpu", "--num_processes", str(len(gpu_ids))]

    if not (task := tm.create_task(args, customize_env)):
        return APIResponse(status="error", message="Failed to create task / 无法创建训练任务")

    def _run():
        try:
            task.execute()
            result = task.communicate()
            if result.returncode != 0:
                log.error(f"Training failed / 训练失败")
            else:
                log.info(f"Training finished / 训练完成")
        except Exception as e:
            log.error(f"An error occurred when training / 训练出现致命错误: {e}")

    coro = asyncio.to_thread(_run)
    asyncio.create_task(coro)

    return APIResponse(status="success", message=f"Training started / 训练开始 ID: {task.task_id}")
