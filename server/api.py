import json
import os
import uuid
from datetime import datetime

from celery.result import AsyncResult
from fastapi import APIRouter
from starlette.requests import Request

from mikazuki.app.models import (APIResponse, APIResponseFail,
                                 TaggerInterrogateRequest)
from mikazuki.log import log
from mikazuki.tagger.interrogator import (available_interrogators)
from server.task.CeleryApp import celery_app
from server.task.CeleryTaskRequest import CeleryTaskRequest as TaskModel
from server.util.OSSManager import oss
from server.util.utils import write_toml_file

router = APIRouter()


@router.get("/task/{task_id}")
def get_result(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    if task_result.ready():
        if task_result.successful():
            result = task_result.get()
            log.info(f"Task {task_id} success {result}")
            return APIResponse(status="success", data=result)
        else:
            # 任务执行过程中出错,跳过本次任务
            log.error(f"Task {task_id} error")
            try:
                return APIResponseFail(message=task_result.get())
            except Exception as e:
                return APIResponseFail(message=str(e))
    else:
        return APIResponse(status="processing")


@router.post("/process_loratrain")
async def processloratrain(request: Request):
    task_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    toml_file = os.path.join(os.getcwd(), f"config", "autosave", f"{timestamp}_{task_id}.toml")
    json_data = await request.body()
    config: dict = json.loads(json_data.decode("utf-8"))
    log.info(f"process_loratrain {task_id} config {config}")
    config['log_prefix']=config['output_name']
    write_toml_file(toml_file, config)

    toml_oss_key = f"snapme/loratrain/config/{timestamp}_{task_id}.toml"
    oss.upload_to_oss(toml_oss_key, toml_file)

    taskConfig = {
        "toml_file": toml_file,
        "toml_ossfile": toml_oss_key,
    }
    from server.task.CeleryTasks import process_loratrain
    celery_task = process_loratrain.delay(
        task_id=task_id,
        taskType="process_loratrain",
        taskConfig=taskConfig,
    )
    log.info(f"process_loratrain delay {celery_task}")
    return APIResponse(status="success", data={"taskId":celery_task.id})


@router.post("/process_dataset")
async def processdataset(req: TaggerInterrogateRequest):
    task_id = str(uuid.uuid4())

    log.info(f"process_dataset {task_id} request {req}")
    if req.path.startswith('oss://'):
        if req.path.endswith('.zip') == False:
            return APIResponseFail(message="oss文件只支持zip格式")

    elif req.path=='ossfile':
        if len(req.images)==0:
            return APIResponseFail(message="图片地址列表不能为空")
        
    else:
        return APIResponseFail(message="oss文件只支持zip格式或者图片地址列表")

    # 设置算法固定值
    req.batch_input_recursive=True
    req.escape_tag=False
    req.replace_underscore=False

    taskConfig = {
        "batch_input_glob": req.path,
        "batch_input_images": req.images,
        "train_repeat":req.train_repeat,
        "batch_input_recursive": req.batch_input_recursive,
        "batch_output_dir": "",
        "batch_output_filename_format": "[name].[output_extension]",
        "batch_output_action_on_conflict": req.batch_output_action_on_conflict,
        "batch_remove_duplicated_tag": True,
        "batch_output_save_json": False,
        "interrogator_model": req.interrogator_model,
        "threshold": req.threshold,
        "additional_tags": req.additional_tags,
        "exclude_tags": req.exclude_tags,
        "sort_by_alphabetical_order": False,
        "add_confident_as_weight": False,
        "replace_underscore": req.replace_underscore,
        "replace_underscore_excludes": req.replace_underscore_excludes,
        "escape_tag": req.escape_tag,
        "unload_model_after_running": True
    }
    from server.task.CeleryTasks import process_dataset
    celery_task = process_dataset.delay(
        task_id=task_id,
        taskType="process_dataset",
        taskConfig=taskConfig,
    )
    log.info(f"process_dataset delay {celery_task}")
    return APIResponse(status="success", data={"taskId":celery_task.id})
