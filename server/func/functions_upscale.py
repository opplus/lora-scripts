import gc
import logging

import cv2
import torch
from celery.utils.log import get_task_logger
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

logger = get_task_logger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_mode(model_name, model_path, device_id):
    # RealESRNet_x4plus
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    # restorer
    upsampler = RealESRGANer(
        scale=netscale,  # 放大倍率，即超分辨率的因子
        model_path=model_path,  # 预训练模型的路径
        dni_weight=None,  # DNI网络的权重，用于控制去噪强度（Denoising Network Integration）
        model=model,  # 输入模型，一般是降噪后的图像
        tile=0,  # 分块大小，即将图像切割成多个小块进行超分辨率
        tile_pad=10,  # 块与块之间的填充大小
        pre_pad=0,  # 预处理时的填充大小
        half=True,  # 是否使用半精度浮点数进行计算，若args.fp32为True则使用半精度，否则使用全精度
        gpu_id=device_id)  # GPU的ID，用于在多GPU环境下指定使用哪个GPU进行计算

    # initialize model
    if device_id:
       logger.info(f"use gpu:{device_id},  {torch.cuda.is_available()}")

    logger.info(f"upscale load_mode device_id:{device_id}, model_path:{model_path}")
    return upsampler


def upscale_image(path, out_path, upsampler, outscale=2):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    output, _ = upsampler.enhance(img, outscale=outscale)
    logger.info(f"upscale_image path:{path}, out_path:{out_path}")
    cv2.imwrite(out_path, output)


def unload_mode(upsampler, device_id):
    try:
        del upsampler
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
       logger.info(f"upsampler unload model error")

