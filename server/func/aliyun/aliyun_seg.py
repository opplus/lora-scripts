import time
from io import BytesIO

import requests
from PIL import Image
from alibabacloud_imageseg20191230.client import Client
from alibabacloud_imageseg20191230.models import (
    SegmentHeadAdvanceRequest,
    SegmentBodyAdvanceRequest,
)
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions

DEFAULT_CONNECT_TIMEOUT = 5000
DEFAULT_READ_TIMEOUT = 10000

fail_retry_count = 3 - 1


def init_aliseg(ak, sk, endpoint, region):
    config = Config(
        # 创建AccessKey ID和AccessKey Secret，请参考https://help.aliyun.com/document_detail/175144.html。
        # 如果您用的是RAM用户的AccessKey，还需要为RAM用户授予权限AliyunVIAPIFullAccess，请参考https://help.aliyun.com/document_detail/145025.html。
        # 从环境变量读取配置的AccessKey ID和AccessKey Secret。运行代码示例前必须先配置环境变量。
        access_key_id=ak,
        access_key_secret=sk,
        # 访问的域名。
        endpoint=endpoint,
        # 访问的域名对应的region
        region_id=region,
        read_timeout=DEFAULT_READ_TIMEOUT,
        connect_timeout=DEFAULT_CONNECT_TIMEOUT,
    )
    return config


def aliseg_base_func(img_path, config, seg_name, do_seg, retry_count=0, get_mask=True, parse_response_func=None,
                     fail_ori=True):
    print(f"enter aliseg:aliseg_base_func {seg_name}  retry_count:{retry_count} <<< {img_path}")
    try:
        with open(img_path, "rb") as img:
            # 初始化Client
            client = Client(config)
            response = do_seg(client, img)
            del client
            print(response.body)
            if parse_response_func is not None:
                return parse_response_func(img_path, response, img, get_mask)
            elements = response.body.data.elements
            if len(elements) < 1:
                print("no elements")
                return None
            image_url = elements[0].image_url
            x1 = elements[0].x
            y1 = elements[0].y
            width = elements[0].width
            height = elements[0].height

            mask = Image.open(BytesIO(requests.get(image_url).content))
            if get_mask:
                img_tmp = Image.open(img_path)
                ori_width, ori_height = img_tmp.size
                img_tmp.close()

                # alpha = mask.getchannel("A")
                # Create a new image for the mask
                # tmp_mask = Image.new("L", mask.size, 0)
                # tmp_mask.paste(alpha, (0, 0), alpha)
                mask = get_whole_mask(mask, x1, y1, width, height, ori_width, ori_height)
                print(f"{seg_name} ali>>> {mask.size}, {(x1, y1, width, height)}  >>> ori >>> {ori_width, ori_height}")

            return mask
    except Exception as error:
        # 获取整体报错信息
        print(error)
        # 如果获取图片失败,重试次数小于3次,递归调用函数重试
        if retry_count < fail_retry_count:
            retry_count += 1
            print(f"{seg_name} 获取图片失败,重试第{retry_count}次")
            time.sleep(1)
            return aliseg_base_func(img_path, config, seg_name, do_seg, retry_count=retry_count, get_mask=get_mask,
                                    fail_ori=fail_ori)
        else:
            print(f"{seg_name} 图像分割失败！返回原图")
            if fail_ori == True:
                return img
            return None


def aliseg_func_head(img_path, config, return_form="mask"):
    print(f"enter aliseg:aliseg_func_head  <<< {img_path}")

    def do_seg(client, img):
        request = SegmentHeadAdvanceRequest()
        request.image_urlobject = img
        request.return_form = return_form
        runtime = RuntimeOptions(
            read_timeout=DEFAULT_READ_TIMEOUT,
            connect_timeout=DEFAULT_CONNECT_TIMEOUT
        )
        return client.segment_head_advance(request, runtime)

    mask = aliseg_base_func(img_path, config, "aliseg_func_head", do_seg=do_seg, retry_count=0,
                            get_mask=True)

    return mask


def aliseg_func_body(img_path, config, return_form="mask"):
    print(f"enter aliseg:aliseg_func_body  <<< {img_path}")

    def do_seg(client, img):
        request = SegmentBodyAdvanceRequest()
        request.image_urlobject = img
        request.return_form = return_form
        runtime = RuntimeOptions(
            read_timeout=DEFAULT_READ_TIMEOUT,
            connect_timeout=DEFAULT_CONNECT_TIMEOUT
        )
        return client.segment_body_advance(request, runtime)

    def parse_response_func(img_path, response, img, get_mask):
        image_url = response.body.data.image_url
        # return the same size image mask
        mask = Image.open(BytesIO(requests.get(image_url).content))
        return mask

    mask = aliseg_base_func(img_path, config, "aliseg_func_body", do_seg=do_seg, retry_count=0,
                            get_mask=True, parse_response_func=parse_response_func)
    return mask


def get_whole_mask(seg_img, x1, y1, width, height, ori_width, ori_height):
    background = Image.new("RGB", (ori_width, ori_height), color="black")
    background.paste(seg_img, (x1, y1, x1 + width, y1 + height))
    return background


def get_img_with_mask(original_image, mask_image):
    mask_image = mask_image.convert("L")
    # 创建一个新的空白图像，用于存放结果，背景可以是透明或任意颜色
    result_image = Image.new("RGB", original_image.size, color="white")
    # 使用paste方法根据mask将原图像的像素粘贴到结果图像上
    result_image.paste(original_image, (0, 0), mask_image)
    return result_image
