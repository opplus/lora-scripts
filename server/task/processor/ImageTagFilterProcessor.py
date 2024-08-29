import json
import logging
import os
import re
from glob import glob
from pathlib import Path

from celery.utils.log import get_task_logger

from server.util.settings import get_dataset_process_config_by_name

tag_escape_pattern = re.compile(r'([\\()])')

logger = get_task_logger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def image_tag_filter_processor(dataset_dir: str):
    tag_filter_config = get_dataset_process_config_by_name("tag")
    filter_tags_keywords_str = tag_filter_config[
        'filter_tags_keywords'] if tag_filter_config is not None and "filter_tags_keywords" in tag_filter_config else ""
    filter_tags_keywords = filter_tags_keywords_str.split(",") if len(filter_tags_keywords_str) > 0 else []
    added_tags_keywords_str = tag_filter_config[
        'added_fixed_tags'] if tag_filter_config is not None and "added_fixed_tags" in tag_filter_config else ""
    added_tags_keywords = added_tags_keywords_str.split(",") if len(added_tags_keywords_str) > 0 else []
    if len(filter_tags_keywords) == 0 or len(added_tags_keywords) == 0:
        logger.info("filter_tags_keywords or added_tags_keywords is empty")
        return
    logger.info(f"find filter_tags_keywords {len(filter_tags_keywords)},details: {filter_tags_keywords_str} ")
    logger.info(f"find added_tags_keywords {len(added_tags_keywords)},details: {added_tags_keywords_str} ")

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

    supported_extensions = [".txt"]

    paths = [
        Path(p)
        for p in glob(dataset_dir, recursive=True)
        if '.' + p.split('.').pop().lower() in supported_extensions
    ]

    logger.info(f'found {len(paths)} tagfile(s)')

    for path in paths:
        # 读取自动识别的tag
        output = []
        if path.is_file():
            output = path.read_text(errors='ignore').strip().split(',')
        logger.info(
            f'read ai tags:{len(output)} from {path}'
        )
        # 标签过滤
        final_tags = []
        filter_tags = []
        if len(filter_tags_keywords) > 0:
            for tag in output:
                filter_flag = False
                for filter_tag in filter_tags_keywords:
                    if filter_tag in tag:
                        filter_tags.append(tag)
                        filter_flag = True
                        continue
                if filter_flag == False:
                    final_tags.append(tag)
        else:
            final_tags = output
        if len(added_tags_keywords) > 0:
            for added_tag in added_tags_keywords:
                for final_tag_0 in final_tags:
                    if added_tag not in final_tag_0:
                        final_tags.append(added_tag)

        logger.info(
            f'filter complete final_tags:{len(final_tags)}, filter_tags:{len(filter_tags)} from {path}, filtered detail {json.dumps(filter_tags)}'
        )

        path.write_text(
            ', '.join(final_tags),
            encoding='utf-8'
        )

    logger.info('标签过滤/追加完成')
