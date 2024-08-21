import os
import shutil
import zipfile

import toml
from PIL import Image


def read_toml_file(file_path):
    with open(file_path, "r") as f:
        data = toml.load(f)
    return data


def write_toml_file(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(toml.dumps(data))


def unzip_file(zip_file_path, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)


def get_zip_file(input_path, result, ignore=[]):
    '''
    递归目录
    :param input_path: 输入路径
    :param result: 列表
    :param ignore: 忽略文件或目录名
    :return:
    '''
    files = os.listdir(input_path)
    for file in files:
        filePath = input_path + '/' + file
        if file in ignore:
            continue
        if os.path.isdir(filePath):
            get_zip_file(filePath, result, ignore)
        else:
            result.append(filePath)


def compress_zip(input_path, output_path, ignore=[]):
    '''
    :param input_path:  输入路径   /app/adminkit
    :param output_path: 输出路径  /app/adminkit.zip
    :param ignore: 忽略文件或目录名
    :return:
    '''
    outdir = os.path.dirname(output_path)
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as f:
        filelists = []
        get_zip_file(input_path, filelists, ignore)
        for file in filelists:
            file = file.replace('\\', '/')
            input_path = input_path.replace('\\', '/')
            f.write(file, file.replace(input_path, ''))
    return output_path


def list_specified_files(directory, extension):
    return [f for f in os.listdir(directory) if f.endswith(extension)]


def delete_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def resize_img(
        input_image, min_side=1024, max_side=1280, mode=Image.BILINEAR, base_pixel_number=64
):
    w, h = input_image.size
    ratio = min(max_side / max(w, h), min_side / min(w, h))
    w_new, h_new = round(ratio * w), round(ratio * h)

    # 确保长宽都是 base_pixel_number 的整数倍
    w_new = (w_new // base_pixel_number) * base_pixel_number
    h_new = (h_new // base_pixel_number) * base_pixel_number

    return input_image.resize([w_new, h_new], mode)
