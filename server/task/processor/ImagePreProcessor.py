import json
import os
import re
import time
from glob import glob
from pathlib import Path

from PIL import Image

from CodeFormer.functions_codeformer import load_face_mode, unload_face_mode, face_enhance
from server.func.aliyun.aliyun_seg import init_aliseg
from server.func.functions_segbody import do_segbody
from server.func.functions_upscale import upscale_image, load_mode, unload_mode
from server.util.settings import get_dataset_process_config_by_name

tag_escape_pattern = re.compile(r'([\\()])')

import logging
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def image_pre_processor(dataset_dir: str, device_id):
    t0 = time.time()
    try:
        face_enhance_config = get_dataset_process_config_by_name("face_enhance")
        upscale_config = get_dataset_process_config_by_name("upscale")
        segbody_config = get_dataset_process_config_by_name("segbody")
        cropface_config = get_dataset_process_config_by_name("cropface")
        if (upscale_config == None or upscale_config['enable_upscale'] == False) and (
                face_enhance_config == None or face_enhance_config['enablce_face_enhance'] == False):
            logger.info("enable_upscale or enablce_face_enhance is disable")
            return
        logger.info(
            f"device_id:{device_id},enablce_face_enhance:{json.dumps(face_enhance_config)},  upscale_config:{json.dumps(upscale_config)}, dataset_dir:{dataset_dir}")

        if not dataset_dir.endswith('*'):
            if not dataset_dir.endswith(os.sep):
                dataset_dir += os.sep
            dataset_dir += '*'

        dataset_dir += '*'

        # get root directory of input glob pattern
        base_dir = dataset_dir.replace('?', '*')
        base_dir = base_dir.split(os.sep + '*').pop(0)

        # check the input directory path
        if not os.path.isdir(base_dir):
            logger.info('input path is not a directory / 输入的路径不是文件夹，终止识别')
            return 'input path is not a directory'

        supported_extensions = [
            e
            for e, f in Image.registered_extensions().items()
            if f in Image.OPEN
        ]

        paths = [
            Path(p)
            for p in glob(dataset_dir, recursive=True)
            if '.' + p.split('.').pop().lower() in supported_extensions
        ]

        logger.info(f'found {len(paths)} images(s)')
        if segbody_config is not None and segbody_config['enable_segbody'] == True:
            seg_body_batch(segbody_config, paths, device_id)

        if cropface_config is not None and cropface_config['enable_cropface'] == True:
            cropface_config(cropface_config, paths, device_id)

        if face_enhance_config is not None and face_enhance_config['enablce_face_enhance'] == True:
            face_enhance_batch(face_enhance_config, paths, device_id)

        if upscale_config is not None and upscale_config['enable_upscale'] == True:
            up_scale_batch(upscale_config, paths, device_id)

    except Exception as e:
        logger.exception(f"image_pre_processor error {dataset_dir}")
    finally:
        t1 = time.time()
        logger.info(f"******图片预处理完成 cost {t1 - t0} seconds for dataset:{dataset_dir}******")


def seg_body_batch(segbody_config, paths, device_id):
    ali_seg_config = init_aliseg(
        segbody_config['ak'],
        segbody_config['sk'],
        segbody_config['endpoint'],
        segbody_config['region'],
    )
    for path in paths:
        t0 = time.time()
        try:
            image = Image.open(path).convert("RGB")
            ori_width, ori_height = image.size
            seged_img = do_segbody(path, ali_seg_config)
            # 还原尺寸
            image = seged_img.resize(size=(ori_width, ori_height), resample=Image.Resampling.BILINEAR)
            logger.info(f'segbody for image {path}')
            image.save(path, quality=95)
        except Exception as e:
            logger.exception(f'${path} seg_body_batch error: {e}')
        finally:
            t1 = time.time()
            logger.info(f"******seg_body_batch cost {t1 - t0} seconds for path:{path}******")


def crop_face_batch(cropface_config, paths, device_id):
    for path in paths:
        t0 = time.time()
        try:
            image = Image.open(path).convert("RGB")
            ori_width, ori_height = image.size
            # 还原尺寸
            logger.info(f'cropface for image {path}')
            image.save(path, quality=95)
        except Exception as e:
            logger.exception(f'${path} crop_face_batch error: {e}')
        finally:
            t1 = time.time()
            logger.info(f"******crop_face_batch cost {t1 - t0} seconds for path:{path}******")


def up_scale_batch(upscale_config, paths, device_id):
    enable_upscale = upscale_config[
        "enable_upscale"] if upscale_config != None and "enable_upscale" in upscale_config else False
    if enable_upscale == False:
        return
    realesrgan_model = upscale_config[
        "realesrgan_model"] if "realesrgan_model" in upscale_config else "RealESRNet_x4plus"
    realesrgan_model_path = upscale_config[
        "realesrgan_model_path"] if "realesrgan_model_path" in upscale_config else "sd-models/RealESRGAN/RealESRNet_x4plus.pth"
    realesrgan_model_path = os.path.join(os.getcwd(), realesrgan_model_path)
    if not os.path.exists(realesrgan_model_path):
        logger.error(f'${realesrgan_model_path} not exist')
        return
    upscale_minsize = upscale_config["upscale_minsize"] if "upscale_minsize" in upscale_config else 1024
    outscale = upscale_config["outscale"] if "outscale" in upscale_config else 2
    max_side = upscale_config["max_side"] if "max_side" in upscale_config else 1024
    min_side = upscale_config["min_side"] if "min_side" in upscale_config else 768
    upsampler = load_mode(model_name=realesrgan_model, model_path=realesrgan_model_path,
                          device_id=device_id)
    for path in paths:
        t0 = time.time()
        try:
            image = Image.open(path).convert("RGB")
            ori_width, ori_height = image.size
            if ori_width < upscale_minsize or ori_height < upscale_minsize:
                logger.info(
                    f'need up_scale [{ori_width},{ori_height}] upscale_minsize:[{upscale_minsize}] for image {path}')
                absolute_path = str(path.resolve())
                # 使用放大模型进行高清放大
                upscale_image(path=absolute_path, out_path=absolute_path, upsampler=upsampler, outscale=outscale)

                image = Image.open(path)
                up_width, up_height = image.size

                # 还原尺寸
                image = image.resize(size=(ori_width, ori_height), resample=Image.Resampling.BILINEAR)
                resized_width, resized_height = image.size
                logger.info(
                    f'resize_img [{up_width},{up_height}] >>> [{resized_width},{resized_height}] for image {path}')
                image.save(path, quality=95)
            # if max(ori_width,ori_height) > max_side or min(ori_width,ori_height) < min_side:
            #      #不在允许尺寸范围内，进行裁剪 TOOD 以人脸为中心裁剪
            #      print(f"不在允许尺寸范围内，进行裁剪 ")

        except Exception as e:
            logger.exception(f'${path} up_scale error: {e}')
        finally:
            t1 = time.time()
            logger.info(f"******up_scale cost {t1 - t0} seconds for path:{path}******")
    if upsampler is not None:
        unload_mode(upsampler, device_id)


def face_enhance_batch(enhance_config, paths, device_id):
    enablce_face_enhance = enhance_config[
        "enablce_face_enhance"] if enhance_config != None and "enablce_face_enhance" in enhance_config else False
    enable_upscale = enhance_config[
        "enable_upscale"] if enhance_config != None and "enable_upscale" in enhance_config else False
    if enablce_face_enhance == False:
        return
    weight = enhance_config["weight"] if "weight" in enhance_config else 0.7
    outscale = enhance_config["outscale"] if "outscale" in enhance_config else 2

    upsampler = None
    if enable_upscale == True:
        realesrgan_model = enhance_config[
            "realesrgan_model"] if "realesrgan_model" in enhance_config else "RealESRNet_x4plus"
        realesrgan_model_path = enhance_config[
            "realesrgan_model_path"] if "realesrgan_model_path" in enhance_config else "sd-models/RealESRGAN/RealESRNet_x4plus.pth"
        realesrgan_model_path = os.path.join(os.getcwd(), realesrgan_model_path)
        if not os.path.exists(realesrgan_model_path):
            logger.error(f'${realesrgan_model_path} not exist')
            return
        upsampler = load_mode(model_name=realesrgan_model, model_path=realesrgan_model_path,
                              device_id=device_id)
    net, face_helper = load_face_mode(device_id)
    for path in paths:
        t0 = time.time()
        try:
            image = Image.open(path).convert("RGB")
            ori_width, ori_height = image.size
            absolute_path = str(path.resolve())
            # 人脸增强
            face_enhance(
                net=net, face_helper=face_helper, face_path=absolute_path, out_path=absolute_path, device_id=device_id,
                w=weight, upsampler=upsampler, outscale=outscale
            )
            image = Image.open(path)
            up_width, up_height = image.size
            # 还原尺寸
            image = image.resize(size=(ori_width, ori_height), resample=Image.Resampling.BILINEAR)
            resized_width, resized_height = image.size
            logger.info(
                f'resize_img [{up_width},{up_height}] >>> [{resized_width},{resized_height}] for image {path}')
            image.save(path, quality=95)
        except Exception as e:
            logger.exception(f'${path} face_enhance error: {e}')
        finally:
            t1 = time.time()
            logger.info(f"******face_enhance cost {t1 - t0} seconds for path:{path}******")
    if upsampler is not None:
        unload_mode(upsampler, device_id)
    if net is not None:
        unload_face_mode(net, face_helper)
