from PIL import Image

from server.func.aliyun.aliyun_seg import aliseg_func_body
from server.util.utils import resize_img


def do_segbody(
        face_path,
        api_seg_config
):
    face_img = Image.open(face_path).convert("RGB")
    ori_width, ori_height = face_img.size
    if ori_width >= 2000 or ori_height >= 2000:
        face_img = resize_img(face_img, max_side=1920, min_side=1024, base_pixel_number=1)
        face_img.save(face_path, quality=95)
    seged_body = aliseg_func_body(face_path, api_seg_config, return_form='whiteBK')
    return seged_body
