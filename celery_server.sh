#!/bin/bash

#queue_priority=${queue_priority:-0}
queue_priority=1
echo 'CELERY_WORKER_C:'[$CELERY_WORKER_C]
echo 'SERVER_IP:'[$SERVER_IP]
echo 'queue_priority:'[$queue_priority]

date_time=`date +"%Y%m%d%H%M"`

#pkill celery

# 设置日志目录路径
log_dir="logs"

# 检查目录是否存在
if [ ! -d "${log_dir}" ]; then
    echo "The directory ${log_dir} does not exist. Creating it now."
    mkdir -p "${log_dir}"
fi

for idx in $(seq 1 $CELERY_WORKER_C); do

  echo "=============================="
  echo 'celery worker服务'${idx}'开始启动。。。'
  if [ "$queue_priority" -eq "1" ]; then
      celery -A server.task.CeleryTasks.celery_app worker -Q h_process_loratrain,process_loratrain,process_dataset -n lora_worker_celery_${SERVER_IP}_${idx}  --loglevel=info --pool=solo --heartbeat-interval=60 --time-limit=1800 >${log_dir}/lora_worker_celery_${SERVER_IP}_log_${idx}.txt 2>&1 &
  else
      celery -A server.task.CeleryTasks.celery_app worker -Q process_loratrain,process_dataset -n lora_worker_celery_${SERVER_IP}_${idx}  --loglevel=info --pool=solo --heartbeat-interval=60 --time-limit=1800 >${log_dir}/lora_worker_celery_${SERVER_IP}_log_${idx}.txt 2>&1 &
  fi
  echo 'celery worker服务'${idx}'启动成功。。。'
  echo "=============================="

done

# celery -A server.task.CeleryTasks.celery_app worker -Q h_process_loratrain,process_loratrain,process_dataset -n lora_worker_celery_xiangongyun_0  --loglevel=info --pool=solo --heartbeat-interval=60 --time-limit=1800 >logs/lora_worker_celery_xiangongyun_log_0.txt 2>&1 &