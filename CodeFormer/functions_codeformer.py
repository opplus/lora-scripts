import gc
import logging
import os

import cv2
import torch
from celery.utils.log import get_task_logger
from torchvision.transforms.functional import normalize

from CodeFormer.basicsr.utils import imwrite, img2tensor, tensor2img
from CodeFormer.basicsr.utils.download_util import load_file_from_url
from CodeFormer.basicsr.utils.registry import ARCH_REGISTRY
from CodeFormer.facelib.utils.face_restoration_helper import FaceRestoreHelper

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

logger = get_task_logger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

face_detection_model = "retinaface_resnet50"
face_detection_model_dir = os.path.join(os.getcwd(), "CodeFormer/weights/CodeFormer")

def load_face_mode(device,model_name="codeformer.pth"):
    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                          connect_list=['32', '64', '128', '256'])

    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'],
                                   model_dir=face_detection_model_dir, progress=True, file_name=model_name)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()
    net.to(f"cuda:{device}")

    # ------------------ set up FaceRestoreHelper -------------------
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'

    face_helper = FaceRestoreHelper(
        2,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=face_detection_model,
        save_ext='png',
        use_parse=True,
        device=f"cuda:{device}")

    logger.info(f"load_face_mode with device_id:{device}")
    return net, face_helper


def unload_face_mode(net, face_helper):
    try:
        del net
        del face_helper
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.exception(f"face_ehance unload model error")


def face_enhance(net, face_helper, face_path, out_path, device_id, w=0.7, upsampler=None, outscale=2):
    face_helper.clean_all()
    img = cv2.imread(face_path, cv2.IMREAD_COLOR)
    face_helper.read_image(img)
    # get face landmarks for each face
    num_det_faces = face_helper.get_face_landmarks_5(only_center_face=True, resize=640, eye_dist_threshold=5)
    logger.info(f'\tdetect {num_det_faces} faces')
    # align and warp each face
    face_helper.align_warp_face()

    # face restoration for each cropped face
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        # prepare data
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(f"cuda:{device_id}")

        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=w, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            logger.info(f'\tFailed inference for CodeFormer: {error}')
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype('uint8')
        face_helper.add_restored_face(restored_face, cropped_face)

    # paste_back

    # upsample the background
    if upsampler is not None:
        # Now only support RealESRGAN for upsampling background
        bg_img = upsampler.enhance(img, outscale=outscale)[0]
    else:
        bg_img = None
    face_helper.get_inverse_affine(None)
    # paste each restored face to the input image
    if upsampler is not None:
        restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=False,
                                                              face_upsampler=upsampler)
    else:
        restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=False)

    # save restored img
    imwrite(restored_img, out_path)
    logger.info(f"face_enhance success >>> {out_path}")
