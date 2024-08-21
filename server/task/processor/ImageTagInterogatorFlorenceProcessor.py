import json
import logging
import os
import time
from glob import glob
from pathlib import Path

import torch
from PIL import Image
from celery.utils.log import get_task_logger
from transformers import AutoProcessor, AutoModelForCausalLM
from mikazuki.tagger import format
from server.util.settings import get_dataset_process_config_by_name

logger = get_task_logger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


model_path = "./sd-models/Florence2/Florence-2-large-ft"

colormap = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'olive', 'cyan', 'red',
            'lime', 'indigo', 'violet', 'aqua', 'magenta', 'gold', 'tan', 'skyblue']

prompts = {
    'region_caption': '<OD>',
    'dense_region_caption': '<DENSE_REGION_CAPTION>',
    'region_proposal': '<REGION_PROPOSAL>',
    'caption': '<CAPTION>',
    'detailed_caption': '<DETAILED_CAPTION>',
    'more_detailed_caption': '<MORE_DETAILED_CAPTION>',
    'caption_to_phrase_grounding': '<CAPTION_TO_PHRASE_GROUNDING>',
    'referring_expression_segmentation': '<REFERRING_EXPRESSION_SEGMENTATION>',
    'ocr': '<OCR>',
    'ocr_with_region': '<OCR_WITH_REGION>',
    'docvqa': '<DocVQA>'
}

batch_output_filename_format = "[name].[output_extension]"


def interogator(dataset_dir: str, additional_tags, device_id):
    try:
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
        t0 = time.time()
        do_interogator(dataset_dir, additional_tags, paths, device_id)
        t1 = time.time()
        logger.info(f"******florence打标完成 cost {t1 - t0} seconds for dataset:{dataset_dir}******")
        return True
    except Exception as e:
        logger.error(f'florence  interogator error: {e}')
        raise e


def do_interogator(dataset_dir: str, additional_tags, paths, device_id):
    processor = ImageTagInterogatorFlorenceProcessor()
    processor.load(device_id=device_id)
    # get root directory of input glob pattern
    base_dir = dataset_dir.replace('?', '*')
    base_dir = base_dir.split(os.sep + '*').pop(0)
    tag_config = get_dataset_process_config_by_name("tag")
    florence_task=tag_config["florence_task"] if "florence_task" in tag_config else "detailed_caption"
    for path in paths:
        t0 = time.time()
        try:
            image = Image.open(path).convert("RGB")

            # guess the output path
            base_dir_last = Path(base_dir).parts[-1]
            base_dir_last_idx = path.parts.index(base_dir_last)
            output_dir = Path(base_dir)
            output_dir = output_dir.joinpath(
                *path.parts[base_dir_last_idx + 1:]).parent

            output_dir.mkdir(0o777, True, True)

            # format output filename
            format_info = format.Info(path, 'txt')
            logger.info(f'format_info {format_info}, batch_output_filename_format:{batch_output_filename_format}')
            formatted_output_filename = format.pattern.sub(
                lambda m: format.format(m, format_info),
                batch_output_filename_format
            )

            output_path = output_dir.joinpath(
                formatted_output_filename
            )
            logger.info(f'output_path {output_path}')

            tag = processor.do_interogator(image,task=florence_task)
            final_tag = f"{additional_tags},{tag}"

            output_path.write_text(
                final_tag,
                encoding='utf-8'
            )
            logger.info(f"florence do_interogator {output_path} <<< {final_tag}")
        except Exception as e:
            logger.error(f'{path} do_interogator error: {e}')
            raise e
        finally:
            t1 = time.time()
            logger.info(f"******do_interogator cost {t1 - t0} seconds for path:{path}******")
    processor.unload()
    del processor


class ImageTagInterogatorFlorenceProcessor():
    def do_interogator(self, image, task='more_detailed_caption'):
        prompt = prompts.get(task, 'more_detailed_caption')
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(torch.float16).to(
            f"cuda:{self.device_id}")
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        logger.info(f"Generated text: {generated_text}")
        parsed_answer = self.processor.post_process_generation(generated_text, task=prompt,
                                                               image_size=(image.width, image.height))
        logger.info(f"parsed_answer text: {parsed_answer}")
        return parsed_answer[prompt]

    def load(self, device_id):
        self.device_id = device_id
        self.model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,trust_remote_code=True)
        self.model.to(f"cuda:{device_id}")
        self.processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
        logger.info(f"load florence model to {device_id} from {model_path}")

    def unload(self):
        del self.model
        del self.processor
        logger.info(f"unload florence model")
