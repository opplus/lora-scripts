import logging
import os
import subprocess
import sys
import time
from abc import abstractmethod
from datetime import datetime

from celery.utils.log import get_task_logger

from mikazuki.tagger.interrogator import on_interrogate, available_interrogators
from mikazuki.utils import train_utils
from server.task.CeleryTaskRequest import CeleryTaskRequest as TaskModel
from server.task.processor.ImagePreProcessor import image_pre_processor
from server.task.processor.ImageTagFilterProcessor import image_tag_filter_processor
from server.task.processor.ImageTagInterogatorFlorenceProcessor import interogator
from server.util.OSSManager import oss
from server.util.settings import get_dataset_process_config_by_name
from server.util.utils import read_toml_file, write_toml_file, unzip_file, compress_zip, list_specified_files, \
    delete_directory

# 设置日志
logger = get_task_logger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

avaliable_scripts = [
    "networks/extract_lora_from_models.py",
    "networks/extract_lora_from_dylora.py",
    "networks/merge_lora.py",
    "tools/merge_models.py",
]

trainer_mapping = {
    "sd-lora": "./sd-scripts/train_network.py",
    "sdxl-lora": "./sd-scripts/sdxl_train_network.py",
    "sd-dreambooth": "./sd-scripts/train_db.py",
    "sdxl-finetune": "./sd-scripts/sdxl_train.py",
}


def dispatch_run(task, environ=None):
    if task.taskType == 'process_loratrain':
        return LoraTrainTaskExecutor().run(task, environ)
    if task.taskType == 'process_dataset':
        return LoraDatasetTaskExecutor().run(task, environ)
    raise RuntimeError(f"unsupport taskType: {task.taskType}, taskId: {task.task_id}")


class TaskExecutor():
    @abstractmethod
    def run(self, task, environ=None):
        pass


def del_file(path):
    try:
        DEL_TEMP_FILE = os.getenv("DEL_TEMP_FILE")
        if DEL_TEMP_FILE and (DEL_TEMP_FILE == 1 or DEL_TEMP_FILE == '1'):
            logger.info(f"del_file  {path}")
            delete_directory(path)
    except Exception as e:
        logger.error(f"del_file error {path}, {e}")


class LoraDatasetTaskExecutor(TaskExecutor):

    def save_dataset_2_local(self, task: TaskModel, base_dir):
        taskConfig = task.taskConfig
        batch_input_glob = taskConfig['batch_input_glob']
        dataset_dir = os.path.join(os.getcwd(), f"dataset", base_dir, "input")
        dataset_dir_zip = os.path.join(os.getcwd(), f"dataset", base_dir, f"input.zip")
        if batch_input_glob.startswith('oss://'):
            batch_input_glob = batch_input_glob.replace('oss://', '')
            # 下载zip文件到本地
            oss.download_from_oss(batch_input_glob, dataset_dir_zip)
            unzip_file(dataset_dir_zip, dataset_dir)
            return dataset_dir
        if batch_input_glob == 'ossfile':
            batch_input_images: list = taskConfig['batch_input_images']
            train_repeat = taskConfig['train_repeat']
            dataset_dir_images = os.path.join(os.getcwd(), f"dataset", base_dir, f"input/{train_repeat}_images")
            os.makedirs(dataset_dir_images, exist_ok=True)
            for oss_file in batch_input_images:
                filename = os.path.basename(oss_file)
                local_file = f'{dataset_dir_images}/{filename}'
                oss.download_from_oss(oss_file, local_file)
                logger.info(f'train_image download_from_oss {oss_file} >>> {local_file}')
            return dataset_dir

    def run(self, task: TaskModel, environ=None):
        date = datetime.now().strftime("%Y%m%d")
        task_id = task.task_id
        taskConfig = task.taskConfig
        base_dir = f"snapme_face/{date}/{task_id}"
        os_base_dir = os.path.join(os.getcwd(), f"dataset", base_dir)
        batch_output_dir_zip = os.path.join(os.getcwd(), f"dataset", base_dir, f"output.zip")
        output_zip_ossfile = f"snapme/loradataset/{date}/{task_id}.zip"

        os.makedirs(os_base_dir, exist_ok=True)

        # 保存dataset到本地
        dataset_dir = self.save_dataset_2_local(task, base_dir)

        allowd_gpu_ids = environ['allowd_gpu_ids']
        CUDA_VISIBLE_DEVICES = environ['CUDA_VISIBLE_DEVICES']
        device_id_arr = allowd_gpu_ids.split(',')
        real_device_id = CUDA_VISIBLE_DEVICES.split(',')[0]
        device_id = 0
        for idx in range(len(device_id_arr)):
            if real_device_id == device_id_arr[idx]:
                device_id = idx
                break

        # 图像预处理
        image_pre_processor(dataset_dir, device_id)

        success_flag=False

        additional_tags = taskConfig['additional_tags']

        tag_config = get_dataset_process_config_by_name("tag")
        tag_type = tag_config["tag_type"] if "tag_type" in tag_config else "florence"
        if tag_type=='florence':
             success_flag=interogator(dataset_dir,additional_tags,device_id)
        else:
            interrogator = available_interrogators.get(taskConfig['interrogator_model'],
                                                       available_interrogators["wd-convnext-v3"])
            ret = on_interrogate(
                image=None,
                batch_input_glob=dataset_dir,
                batch_input_recursive=taskConfig['batch_input_recursive'],
                batch_output_dir=taskConfig['batch_output_dir'],
                batch_output_filename_format=taskConfig['batch_output_filename_format'],
                batch_output_action_on_conflict=taskConfig['batch_output_action_on_conflict'],
                batch_remove_duplicated_tag=taskConfig['batch_remove_duplicated_tag'],
                batch_output_save_json=taskConfig['batch_output_save_json'],
                interrogator=interrogator,
                threshold=taskConfig['threshold'],
                additional_tags=taskConfig['additional_tags'],
                exclude_tags=taskConfig['exclude_tags'],
                sort_by_alphabetical_order=taskConfig['sort_by_alphabetical_order'],
                add_confident_as_weight=taskConfig['add_confident_as_weight'],
                replace_underscore=taskConfig['replace_underscore'],
                replace_underscore_excludes=taskConfig['replace_underscore_excludes'],
                escape_tag=taskConfig['escape_tag'],
                unload_model_after_running=taskConfig['unload_model_after_running']
            )
            # 标签过滤
            image_tag_filter_processor(dataset_dir)
            if "Succeed" == ret:
                success_flag=True



        if success_flag==True:
            logger.info("on_interrogate success")
            compress_zip(dataset_dir, batch_output_dir_zip)
            logger.info(f"on_interrogate compress_zip  {batch_output_dir_zip}")
            retry_num=10
            while retry_num>0:
                try:
                    oss.upload_to_oss(output_zip_ossfile, batch_output_dir_zip)
                    logger.info(f"on_interrogate upload_to_oss {output_zip_ossfile}")
                    break
                except Exception as e:
                    logger.exception(f"upload_to_oss error {output_zip_ossfile}")
                    time.sleep(2)
                    retry_num = retry_num - 1

            # del_file(os_base_dir)
            return {"status": "success", "data": {"output_zip_ossfile": output_zip_ossfile}}
        logger.info(f"on_interrogate error")
        return {"status": "fail", "message": "error"}


class LoraTrainTaskExecutor(TaskExecutor):

    def run(self, task: TaskModel, environ=None):
        t0 = time.time()
        task_id = task.task_id
        taskConfig = task.taskConfig
        toml_ossfile = taskConfig['toml_ossfile']
        date = datetime.now().strftime("%Y%m%d")
        base_dir = f"config/snapme_face/{date}/{task_id}"
        oss_toml_path = os.path.join(os.getcwd(), base_dir, f"oss.toml")
        local_toml_path = os.path.join(os.getcwd(), base_dir, f"local.toml")
        train_data_dir = os.path.join(os.getcwd(), base_dir, "dataset")
        train_data_dir_zip = os.path.join(os.getcwd(), base_dir, f"dataset.zip")

        os.makedirs(train_data_dir, exist_ok=True)

        # 下载toml文件到本地
        oss.download_from_oss(toml_ossfile, oss_toml_path)
        config: dict = read_toml_file(oss_toml_path)
        train_data_ossfile = config["train_data_ossfile"]
        model_train_type = config["model_train_type"]

        # retake_white_1
        output_name = config["output_name"]
        output_dir_prefix = f"{date}/{task_id}/{output_name}"
        # ./output/{output_dir_prefix}
        output_dir = f"./output/{output_dir_prefix}"
        config["output_dir_prefix"] = output_dir_prefix
        config["output_dir"] = output_dir

        os_base_dir = os.path.join(os.getcwd(), base_dir)
        os_output_dir = os.path.join(os.getcwd(), f"output/{output_dir_prefix}")

        trainer_file = trainer_mapping[model_train_type]
        validated, message = train_utils.validate_model(config["pretrained_model_name_or_path"],model_train_type)
        if not validated:
            raise RuntimeError(message)

        # 下载预览prompt到本地
        sample_prompts = config.get("sample_prompts")
        if sample_prompts is not None and not os.path.exists(sample_prompts) and train_utils.is_promopt_like(
                sample_prompts):
            sample_prompts_file = os.path.join(os.getcwd(), base_dir, f"sample_promopt.txt")
            with open(sample_prompts_file, "w", encoding="utf-8") as f:
                f.write(sample_prompts)
            config["sample_prompts"] = sample_prompts_file
            logger.info(f"Wrote promopts to file {sample_prompts_file}")

        # 下载训练集到本地
        oss.download_from_oss(train_data_ossfile, train_data_dir_zip)
        unzip_file(train_data_dir_zip, train_data_dir)

        # 写入配置文件到本地
        config['train_data_dir'] = train_data_dir
        write_toml_file(local_toml_path, config)

        suggest_cpu_threads = 8 if len(train_utils.get_total_images(train_data_dir)) > 200 else 2

        args = [
            sys.executable, "-m", "accelerate.commands.launch",  # use -m to avoid python script executable error
            "--num_cpu_threads_per_process", str(suggest_cpu_threads),  # cpu threads
            "--quiet",  # silence accelerate error message
            trainer_file,
            "--config_file", local_toml_path,
        ]
        logger.info(f"subprocess args:{args}, environ:{environ}")
        process = subprocess.Popen(args, env=environ)

        try:
            stdout, stderr = process.communicate()
            logger.info(f"stdout:{stdout}")
            logger.info(f"stderr:{stderr}")
        except:
            process.kill()
            raise
        retcode = process.poll()
        t1 = time.time()
        train_cost = t1 - t0
        logger.info("******train time {:.2f} seconds******".format(train_cost))
        if retcode != 0:
            logger.error(f"Training failed")
            del_file(os_base_dir)
            del_file(os_output_dir)
            return {"status": "fail", "cost": train_cost}
        else:
            logger.info(f"Training finished")
            # 增加训练成功后处理， 把训练成功的lora文件上传到oss
            train_result = self.build_train_result(config)
            t2 = time.time()
            total_cost = t2 - t0
            result_cost = t2 - t1
            del_file(os_base_dir)
            # del_file(os_output_dir)
            logger.info("******train time total {:.2f} seconds******".format(total_cost))
            return {
                "status": "success",
                "data": train_result,
                "cost": total_cost,
                "train_cost": train_cost,
                "result_cost": result_cost
            }

    def build_train_result(self, trainConfig: dict):
        # retake_white_1
        output_name = trainConfig["output_name"]
        save_model_as = trainConfig["save_model_as"]
        # {date}/{taskId}/{output_name}
        output_dir_prefix = trainConfig["output_dir_prefix"]
        max_train_epochs = trainConfig["max_train_epochs"]
        # 列出目录下的文件
        output_dir = os.path.join(os.getcwd(), f"output", output_dir_prefix)
        lora_files = list_specified_files(output_dir, f'.{save_model_as}')

        sample_dir = os.path.join(os.getcwd(), output_dir, "sample")
        sample_files = []
        if os.path.exists(sample_dir):
            sample_files = list_specified_files(sample_dir, f'.png')

        lora_path_prefix = os.getenv("LORA_PATH_PREFOX")
        if not lora_path_prefix.endswith("/"):
            lora_path_prefix = lora_path_prefix + "/"
        loras = []
        for lora_file in lora_files:
            lora_file_name = lora_file.split("/")[-1]  # retake_white_1-000002.safetensors
            lora_path = f'{lora_path_prefix}{output_dir_prefix}/{lora_file_name}'  # {date}/{taskId}/retake_white_1/retake_white_1-000002.safetensors
            sample_file_name = None
            sample_file_osspath = ""
            # 样例图片处理
            try:
                lora_epoch_name = lora_file_name.split(".")[0]  # retake_white_1-000002,retake_white_1
                logger.info(f'lora_epoch_name:{lora_epoch_name}')
                name_arr = lora_epoch_name.rsplit('-', 1)  # [retake_white_1,000002]
                if len(name_arr) == 2:
                    sample_file_name_prefix = f"{name_arr[0]}_e{name_arr[1]}"  # retake_white_1_e000002
                else:
                    sample_file_name_prefix = f"{name_arr[0]}_e{max_train_epochs:06d}"  # retake_white_1_e000002
                logger.info(f'sample_file_name_prefix:{sample_file_name_prefix}, name_arr:{name_arr}')
                # {date}/{taskId}/retake_white_1/sample/retake_white_1_e000002_00_20240612161701.png
                for f in sample_files:
                    if f.startswith(sample_file_name_prefix):
                        sample_file_name = f  # retake_white_1_e000002_00_20240612161701.png
                        break
                if sample_file_name is not None:
                    local_sample_file_path = os.path.join(os.getcwd(), f"output", output_dir_prefix, 'sample',
                                                          sample_file_name)
                    sample_file_osspath = f"snapme/lora/{output_dir_prefix}/sample/{sample_file_name}"
                    logger.info(
                        f'local_sample_file_path:{local_sample_file_path} >>> sample_file_osspath:{sample_file_osspath}')
                    oss.upload_to_oss(sample_file_osspath, local_sample_file_path)
            except Exception as e:
                logger.error(f"upload sample_image:{sample_file_name} to oss failed due to an exception.{e}")

            loras.append({
                "path": f'{lora_path}',
                "sample": sample_file_osspath
            })

        train_result = {
            "loras": loras,
            "count": len(lora_files),
            "loraName": output_name
        }
        return train_result
