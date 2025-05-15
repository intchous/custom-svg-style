import os
import random
import cv2
import numpy as np
import re
import pandas as pd
import yaml
from box import Box
import gc

import torch
import PIL
from PIL import Image
from diffusers.utils import make_image_grid
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionPipeline, DiffusionPipeline, StableDiffusionControlNetPipeline
from diffusers import ControlNetModel, EulerAncestralDiscreteScheduler
from controlnet_aux import HEDdetector, CannyDetector, NormalBaeDetector, MLSDdetector, PidiNetDetector

from svg_ldm.dreambooth.test_dreambooth_lora_forimgs import load_sd_lora
from svg_ldm.dreambooth.get_img_cids import get_img_cid_list
from svg_ldm.prompts_list import get_prompt_description_vecdiffusion_list


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


def prepare_image_cond(cond_rgb, control_type="normal", canny_lower_bound=50, canny_upper_bound=100, blur_ksize=1, device="cuda"):
    # cond_rgb: Float[Tensor, "B C H W"]
    cond_rgb_bhwc = cond_rgb.permute(0, 2, 3, 1)

    if control_type == "normal":
        preprocessor = NormalBaeDetector.from_pretrained(
            "lllyasviel/Annotators")
        preprocessor.model.to(device)

        cond_rgb_hwc = (
            (cond_rgb_bhwc[0].detach().cpu().numpy()
                * 255).astype(np.uint8).copy()
        )
        detected_map = preprocessor(cond_rgb_hwc)
        detected_map = np.array(detected_map)

        control = (torch.from_numpy(
            np.array(detected_map)).float().to(device) / 255.0)

        control = control.unsqueeze(0)
        control = control.permute(0, 3, 1, 2)

    elif control_type == "canny":
        preprocessor = CannyDetector()

        cond_rgb_hwc = (
            (cond_rgb_bhwc[0].detach().cpu().numpy()
             * 255).astype(np.uint8).copy()
        )
        if (blur_ksize > 1):
            blurred_img = cv2.blur(
                cond_rgb_hwc, ksize=(blur_ksize, blur_ksize))
        else:
            blurred_img = cond_rgb_hwc

        detected_map = preprocessor(
            blurred_img, canny_lower_bound, canny_upper_bound)

        control = (torch.from_numpy(
            np.array(detected_map)).float().to(device) / 255.0)

        control = control.unsqueeze(0)
        control = control.permute(0, 3, 1, 2)

    # control: Float[Tensor, "B C H W"]
    return control


def prepare_image_cond_sd15(original_image, control_type="canny", low_threshold=50, high_threshold=100, blur_ksize=0):

    if "canny" in control_type:
        np_image = np.array(original_image)
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        kernel = np.ones((3, 3), np.uint8)
        dilation_iter = 1
        erosion_iter = 1
        edges = cv2.dilate(edges, kernel, iterations=dilation_iter)
        edges = cv2.erode(edges, kernel, iterations=erosion_iter)

        if blur_ksize > 0:
            edges = cv2.GaussianBlur(edges, (blur_ksize, blur_ksize), 0)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        control_image = Image.fromarray(edges_3ch)
        # -----------------------------------------

    elif "scribble" in control_type:
        processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
        control_image = processor(original_image, scribble=True)
        # -----------------------------------------

    elif "hed" in control_type:
        processor = HEDdetector.from_pretrained('lllyasviel/Annotators')
        control_image = processor(original_image)

    elif "mlsd" in control_type:
        processor = MLSDdetector.from_pretrained('lllyasviel/Annotators')
        control_image = processor(original_image)

    elif "softedge" in control_type:
        processor = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
        control_image = processor(original_image, safe=True)

    elif "normal" in control_type:
        preprocessor = NormalBaeDetector.from_pretrained(
            "lllyasviel/Annotators")
        control_image = preprocessor(original_image)

    return control_image


def load_target_whitebg(fp, img_size=64):

    target = PIL.Image.open(fp)

    if target.size != (img_size, img_size):
        target = target.resize((img_size, img_size),
                               PIL.Image.Resampling.BICUBIC)

    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = PIL.Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    # target = np.array(target)
    return target


def load_sd15_controlnet(pretrained_model_name="runwayml/stable-diffusion-v1-5", controlnet_id="lllyasviel/sd-controlnet-canny", clip_skip=0):

    controlnet = ControlNetModel.from_pretrained(
        controlnet_id, torch_dtype=torch.float16)

    pipe_test = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained_model_name,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        # torch_dtype=torch.float32,
    ).to("cuda")

    pipe_test.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe_test.scheduler.config)

    pipe_test.safety_checker = lambda clip_input, images: (
        images, [False for _ in images])

    # clip_skip=2
    if clip_skip > 0:
        clip_layers = pipe_test.text_encoder.text_model.encoder.layers
        pipe_test.text_encoder.text_model.encoder.layers = clip_layers[:-clip_skip]

    # pipe_test.enable_model_cpu_offload()
    return pipe_test


def load_img2img_sd15_controlnet(pretrained_model_name="runwayml/stable-diffusion-v1-5", controlnet_id="lllyasviel/sd-controlnet-canny", clip_skip=0):

    controlnet = ControlNetModel.from_pretrained(
        controlnet_id, torch_dtype=torch.float16)

    pipe_test = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        pretrained_model_name,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        # torch_dtype=torch.float32,
    ).to("cuda")

    pipe_test.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe_test.scheduler.config)

    pipe_test.safety_checker = lambda clip_input, images: (
        images, [False for _ in images])

    # clip_skip=2
    if clip_skip > 0:
        clip_layers = pipe_test.text_encoder.text_model.encoder.layers
        pipe_test.text_encoder.text_model.encoder.layers = clip_layers[:-clip_skip]

    # pipe_test.enable_model_cpu_offload()
    return pipe_test


def prompt_template(caption, instance_prompt="ICF v3ct0r"):
    prompt_appd = "white background."
    if (instance_prompt != ""):
        prompt = f'A flat vector icon of {caption}, in {instance_prompt} style' + \
            ", " + prompt_appd
    else:
        prompt = "A flat vector icon of " + caption + ", " + prompt_appd
    print("prompt:", prompt)

    negative_prompt = "low quality, blurry, noise, watermark"

    return prompt, negative_prompt


def format_filename(filename):
    # 使用正则表达式匹配"rand"开头和"_数字.svg"结尾的文件名
    match = re.match(r"^rand\d+_(.*?)_\d+\.png$", filename)
    prompt = ""
    if match:
        # 提取中间部分并替换下划线为", "
        core = match.group(1)
        prompt = ", ".join(core.split("_"))

    return prompt


def get_id_label_map(caption_fp, signature="iconfont_collections"):
    caption_df = pd.read_csv(caption_fp)

    # 生成 id-label 的映射字典
    if ("IconShop" in signature):
        id_label_map = dict(
            zip(caption_df['id'].astype(int), caption_df['label'].astype(str)))
    else:
        id_label_map = dict(zip(caption_df['id'].astype(
            str), caption_df['label'].astype(str)))

    return id_label_map


def get_caption(cur_id, id_label_map, signature="iconfont_collections"):
    if ("IconShop" in signature):
        label = id_label_map.get(int(cur_id), "")
    else:
        label = id_label_map.get(str(cur_id), "")
    label = label.replace("/", ", ")
    # assert (label != "")

    return label


def main():

    # CUDA_VISIBLE_DEVICES=1 python -m svg_ldm.img2img.controlnet_sd15

    yaml_fp = os.path.join(
        "./svg_ldm/img2img/control_config/controlnet_sd15.yaml")
    with open(yaml_fp, 'r') as f:
        config_data = yaml.safe_load(f)

    # 将字典转换为 AttrDict 对象
    cfg = Box(config_data)

    # -----------------------------------------
    dataset_sign = "svgrepo_collections"

    par_dir = os.path.join("./dataset/", dataset_sign)
    all_imgs_data_dir = os.path.join(par_dir, f"{dataset_sign}_diffvg_select")

    caption_fp = os.path.join(par_dir, "label.csv")
    id_label_map = get_id_label_map(caption_fp, signature=dataset_sign)

    # -----------------------------------------
    num_images_per_prompt = 10

    num_inference_steps = 30
    controlnet_conditioning_scale = 0.5
    guidance_scale = 7.0
    clip_skip = 2
    strength = 1.0  # 0.75
    blur_radius = 7

    num_inference_steps = 50
    strength = 1.0

    controlnet_conditioning_scale = 0.5
    # -----------------------------------------

    signature = cfg.signature
    instance_prompt = cfg.instance_prompt
    num_inference_steps = cfg.base_steps
    # normal, canny, scribble
    control_type = cfg.control_type

    lora_scale_unet = cfg.lora_scale_unet
    lora_scale_text_encoder = cfg.lora_scale_text_encoder
    guidance_scale = cfg.base_scale

    condition_scale = cfg.condition_scale
    strength = cfg.strength

    low_threshold = 50 * 1
    high_threshold = 100 * 1

    log_dir = "./dreamb_logs/"

    if ("img2img" in control_type):
        control_desc = "controlnet_sd15_" + "dt-" + dataset_sign + \
            "_" + control_type + '_conds' + \
            str(condition_scale) + '_s' + str(strength)
    else:
        control_desc = "controlnet_sd15_" + "dt-" + dataset_sign + \
            "_" + control_type + '_conds' + str(condition_scale)
    print("control_desc: ", control_desc)

    desc = "dreamb_lora_" + "dt-" + signature + "_pty-" + \
        str(cfg.instance_prompt_type) + "_rank-" + str(cfg.lora_rank)
    print("lora_desc: ", desc)

    output_dir = os.path.join(log_dir, "dream_control", control_desc, desc)
    os.makedirs(output_dir, exist_ok=True)

    if ("canny" in control_type):
        controlnet_id = "lllyasviel/sd-controlnet-canny"
    elif ("scribble" in control_type):
        controlnet_id = "lllyasviel/sd-controlnet-scribble"
    elif ("hed" in control_type):
        controlnet_id = "lllyasviel/sd-controlnet-hed"
    elif ("mlsd" in control_type):
        controlnet_id = "lllyasviel/control_v11p_sd15_mlsd"
    elif ("softedge" in control_type):
        controlnet_id = "lllyasviel/control_v11p_sd15_softedge"
    elif ("normal" in control_type):
        controlnet_id = "lllyasviel/control_v11p_sd15_normalbae"

    # -----------------------------------------
    model_par_dir = os.path.join(log_dir, "models", desc)

    img_cid_list = get_img_cid_list()
    print("len(img_cid_list): ", len(img_cid_list))

    caption_list_train = get_prompt_description_vecdiffusion_list()
    print("len(caption_list_train): ", len(caption_list_train))

    random.shuffle(img_cid_list)
    for tmp_cid in img_cid_list:
        if (tmp_cid == ".DS_Store" or tmp_cid == ".ipynb_checkpoints"):
            continue

        cid = str(tmp_cid)
        model_save_dir = os.path.join(model_par_dir, cid)
        if (not os.path.isdir(model_save_dir)):
            continue

        print("cid: ", cid)

        # -----------------------------------------
        st_md_num = 1000
        ed_md_num = 2000

        model_list = os.listdir(model_save_dir)
        for md_fn in model_list:
            if (md_fn == "logs"):
                continue

            process_flag = False
            if md_fn == "pipeline":
                process_flag = True
            elif md_fn.startswith("checkpoint-") and md_fn.split("-")[1].isdigit():
                checkpoint_num = int(md_fn.split("-")[1])
                if st_md_num <= checkpoint_num <= ed_md_num:
                    process_flag = True

            if not process_flag:
                continue

            cur_model_dir = os.path.join(model_save_dir, md_fn)
            dream_pred_dir = os.path.join(output_dir, cid, md_fn)
            os.makedirs(dream_pred_dir, exist_ok=True)

            if ("img2img" in control_type):
                pipe_test = load_img2img_sd15_controlnet(
                    controlnet_id=controlnet_id)
            else:
                pipe_test = load_sd15_controlnet(controlnet_id=controlnet_id)

            pipe_test = load_sd_lora(pipe_test, lora_id=cur_model_dir,
                                     lora_scale_unet=lora_scale_unet, lora_scale_text_encoder=lora_scale_text_encoder)
            # -----------------------------------------
            # processor = Processor('canny')
            test_num = 30
            tmp_test_num = 0
            targets = os.listdir(all_imgs_data_dir)
            random.shuffle(targets)

            for target in targets:
                if (tmp_test_num > test_num):
                    break

                if (not target.endswith('.png')):
                    continue

                # if ("img2img" in target):
                #     continue

                target_pre = os.path.splitext(target)[0]
                target_pre = target_pre.replace("_diffvg_rsz", "")

                target_img_fp = os.path.join(all_imgs_data_dir, target)
                try:
                    original_image = load_target_whitebg(
                        target_img_fp, img_size=512)
                except:
                    print("Can't load target_img_fp: ", target_img_fp)
                    continue

                caption = get_caption(
                    cur_id=target_pre, id_label_map=id_label_map, signature=dataset_sign)

                do_img2img_flg = False
                for simp_caption in caption_list_train:
                    if (simp_caption in caption):
                        do_img2img_flg = True
                        break

                if (not do_img2img_flg):
                    continue

                tmp_test_num += 1
                print('target:', target)

                prompt, negative_prompt = prompt_template(
                    caption, instance_prompt=instance_prompt)

                control_image = prepare_image_cond_sd15(
                    original_image, control_type=control_type, low_threshold=low_threshold, high_threshold=high_threshold, blur_ksize=0)

                if ("img2img" in control_type):
                    output_images = pipe_test(
                        prompt,
                        negative_prompt=negative_prompt,
                        image=original_image,
                        control_image=control_image,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        num_images_per_prompt=num_images_per_prompt,
                        # controlnet_conditioning_scale=controlnet_conditioning_scale,
                        controlnet_conditioning_scale=condition_scale,
                        strength=strength,  # img2img
                    ).images

                else:
                    output_images = pipe_test(
                        prompt,
                        negative_prompt=negative_prompt,
                        controlnet_conditioning_scale=condition_scale,
                        image=control_image,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        num_images_per_prompt=num_images_per_prompt,
                    ).images
                # --------------------------------------------------

                for i, output_sd in enumerate(output_images):
                    output_sd = output_sd.resize(
                        (512, 512), PIL.Image.Resampling.BICUBIC)
                    output_sd.save(os.path.join(
                        dream_pred_dir, f"{target_pre}_img2img_{i+1}.png"))

                    output_images[i] = output_sd

                original_image = original_image.resize(
                    (512, 512), PIL.Image.Resampling.BICUBIC)
                control_image = control_image.resize(
                    (512, 512), PIL.Image.Resampling.BICUBIC)

                output_grid = make_image_grid(
                    [original_image, control_image, *output_images], rows=4, cols=3)
                output_grid.save(
                    f"{dream_pred_dir}/{target_pre}_output_grid.png")

                original_image.save(f"{dream_pred_dir}/{target_pre}.png")
                # -----------------------------------------

            # free some memory
            del pipe_test
            gc.collect()
            torch.cuda.empty_cache()
            # -----------------------------------------


if __name__ == "__main__":
    # set_seed(0)
    main()
