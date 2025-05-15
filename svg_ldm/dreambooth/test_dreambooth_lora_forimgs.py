#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import gc
import os
from pathlib import Path
import yaml
from box import Box
import random

import numpy as np
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
import PIL
from PIL import Image
from torchvision import transforms

from diffusers import StableDiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DDIMScheduler, LMSDiscreteScheduler

from svg_ldm.ldm_dataset_utils import get_data_imgs_list
from svg_ldm.prompts_list import get_prompt_description_vecdiffusion_list
from svg_ldm.dreambooth.get_img_cids import get_img_cid_list


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.33.0.dev0")
logger = get_logger(__name__)


# -----------------------------------------------------
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def remove_background(img_pil, out_fp):
    # u2net, isnet-general-use, isnet-anime, sam
    model_name = "sam"
    session = new_session(model_name)
    img_fg_mask = remove(img_pil, only_mask=True)

    # 0-255
    img_fg_mask_norm = np.array(img_fg_mask, dtype=np.float32) / 255.0

    img_fg_mask_norm[img_fg_mask_norm < 0.5] = 0
    img_fg_mask_norm[img_fg_mask_norm >= 0.5] = 1
    img_fg_mask_ini = img_fg_mask_norm.astype(np.uint8)

    # Expand the mask and copy it to all 3 channels
    img_fg_mask = img_fg_mask_ini[:, :, np.newaxis]
    img_fg_mask = np.repeat(img_fg_mask, 3, axis=2)

    img = np.array(img_pil)
    img_fg = img * img_fg_mask

    # add alpha channel to img_fg
    img_alpha = np.concatenate(
        [img_fg, img_fg_mask_ini[:, :, np.newaxis] * 255], axis=2)
    img_with_alpha = PIL.Image.fromarray(
        img_alpha.astype(np.uint8), mode='RGBA')
    img_with_alpha.save(out_fp)

    return img_with_alpha


def dream_infer(pipe_test, prompt,  num_samples, base_negprompt="", base_steps=35, base_scale=7.5):
    all_images = []
    img_loss_pairs = []
    images = pipe_test(prompt,
                       negative_prompt=base_negprompt,
                       num_images_per_prompt=num_samples,
                       num_inference_steps=base_steps,
                       guidance_scale=base_scale).images

    for i, ini_img in enumerate(images):
        img = ini_img
        t_transform = transforms.ToTensor()
        img_tensor = t_transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_loss_pairs.append((ini_img, 1.0))

    img_loss_pairs.sort(key=lambda x: x[1])
    sorted_images = [pair[0] for pair in img_loss_pairs]
    all_images.extend(sorted_images)

    return all_images


def load_sd_model(pretrained_model_name="runwayml/stable-diffusion-v1-5", scheduler_name="Euler Ancestral"):

    pipe_test = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name,
        torch_dtype=torch.float32,
    ).to("cuda")

    scheduler_mapping = {
        "Euler Ancestral": EulerAncestralDiscreteScheduler,
        "DPM++ 2M Karras": DPMSolverMultistepScheduler,
        "DDIM": DDIMScheduler,
        "LMS": LMSDiscreteScheduler,
    }

    if scheduler_name not in scheduler_mapping:
        raise ValueError(
            f"Unsupported scheduler_name: {scheduler_name}. Supported schedulers: {list(scheduler_mapping.keys())}")

    pipe_test.scheduler = scheduler_mapping[scheduler_name].from_config(
        pipe_test.scheduler.config)

    pipe_test.safety_checker = lambda clip_input, images: (
        images, [False for _ in images])

    return pipe_test


def load_sd_lora(pipe_test, lora_id=None, lora_scale_unet=0.8, lora_scale_text_encoder=0.5):

    # 分别加载 Text Encoder 和 UNet 的 LoRA 权重
    pipe_test.load_lora_weights(
        pretrained_model_name_or_path_or_dict=lora_id,
        text_encoder_only=True,
        lora_scale=lora_scale_text_encoder
    )
    pipe_test.load_lora_weights(
        pretrained_model_name_or_path_or_dict=lora_id,
        unet_only=True,
        lora_scale=lora_scale_unet
    )

    return pipe_test


def dream_infer_multi(model_dir, dream_pred_dir, caption_list_train, pretrained_model_name="runwayml/stable-diffusion-v1-5", lora_scale_unet=0.8, lora_scale_text_encoder=0.5, instance_prompt="ICF v3ct0r style", scheduler_name="Euler Ancestral", base_steps=35, base_scale=7.5, num_prompt_samp=16, num_samples=3, num_rows=2):

    pipe_test = load_sd_model(
        pretrained_model_name=pretrained_model_name, scheduler_name=scheduler_name)

    lora_weights_path = os.path.join(
        model_dir, "pytorch_lora_weights.safetensors")
    pipe_test = load_sd_lora(pipe_test=pipe_test, lora_id=lora_weights_path,
                             lora_scale_unet=lora_scale_unet, lora_scale_text_encoder=lora_scale_text_encoder)

    # -----------------------------------------
    base_negprompt = "ugly, disfigured, poorly drawn face, skin blemishes, skin spots, acnes, missing limb, malformed limbs, floating limbs, disconnected limbs, extra limb, extra arms, mutated hands, poorly drawn hands, malformed hands, mutated hands and fingers, bad hands, missing fingers, fused fingers, too many fingers, extra legs, bad feet, cross-eyed"
    base_negprompt = "low quality, blurry, noise, watermark"

    num_prompt_samp = min(len(caption_list_train), int(num_prompt_samp))
    samp_prompts = random.sample(caption_list_train, num_prompt_samp)

    for _prompt_description in samp_prompts:
        description_start = instance_prompt
        prompt_appd = "white background."

        prompt_description = _prompt_description
        prompt = "A flat vector icon of a " + prompt_description + \
            ", in " + description_start + " style; " + prompt_appd
        print("prompt = ", prompt)

        all_images = []
        num_remain = int(num_samples)
        for sd_i in range(num_rows):
            tmp_images = dream_infer(pipe_test=pipe_test, prompt=prompt,
                                     num_samples=num_samples, base_negprompt=base_negprompt, base_steps=base_steps, base_scale=base_scale)

            all_images.extend(tmp_images[:num_remain])

        start_number = random.randint(10000, 99999)
        for i, img in enumerate(all_images):
            tmp_fp = os.path.join(
                dream_pred_dir, f"{prompt_description}_{start_number + i}.png")
            img.save(tmp_fp)

    # -----------------------------------------
    # free some memory
    del pipe_test
    gc.collect()
    torch.cuda.empty_cache()
    # -----------------------------------------


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=2 python -m svg_ldm.dreambooth.test_dreambooth_lora_forimgs

    parser = argparse.ArgumentParser(description="Load config from yaml file")
    parser.add_argument("--yaml_fn", type=str, default="test_dreambooth_lora_forimgs",
                        help="Path name of the yaml config file")
    args = parser.parse_args()

    yaml_fp = os.path.join(
        "./svg_ldm/dreambooth/db_config/", args.yaml_fn + ".yaml")

    # 从配置文件中加载配置数据
    with open(yaml_fp, 'r') as f:
        config_data = yaml.safe_load(f)

    cfg = Box(config_data)

    # -----------------------------------------
    signature = cfg.signature

    use_svgrepo_prompts = True
    if (use_svgrepo_prompts):
        dataset_sign = "svgrepo_collections"
        par_dir = os.path.join("./dataset/", dataset_sign)
        all_imgs_data_dir = os.path.join(
            par_dir, f"{dataset_sign}_diffvg_select")

        imgs_list_train, caption_list_train = get_data_imgs_list(
            signature=dataset_sign, imgs_data_dir=all_imgs_data_dir, par_dir=par_dir, check_empty=True)

    else:
        par_dir = os.path.join("./dataset/", signature)
        all_imgs_data_dir = os.path.join(par_dir, f"{signature}_diffvg_select")
        imgs_data_par_dir = os.path.join(
            par_dir, f"{signature}_diffvg_select_dreambooth")

        imgs_list_train, caption_list_train = get_data_imgs_list(
            signature=signature, imgs_data_dir=all_imgs_data_dir, par_dir=par_dir, check_empty=True)

    print("len(imgs_list_train): ", len(imgs_list_train))
    desc = "dreamb_lora_" + "dt-" + signature + "_pty-" + \
        str(cfg.instance_prompt_type) + "_rank-" + str(cfg.lora_rank)
    print("desc: ", desc)

    log_dir = "./dreamb_logs/"
    dream_pred_sign = "dream_pred_vecdiff_selectcid"
    dream_pred_par_dir = os.path.join(log_dir, dream_pred_sign, desc)
    os.makedirs(dream_pred_par_dir, exist_ok=True)

    model_par_dir = os.path.join(log_dir, "models", desc)

    # -----------------------------------------
    img_cid_list = get_img_cid_list()
    caption_list_train = get_prompt_description_vecdiffusion_list()
    caption_list_train = list(set(caption_list_train))
    print("len(caption_list_train): ", len(caption_list_train))

    random.shuffle(img_cid_list)
    for tmp_cid in img_cid_list:
        if (tmp_cid == ".DS_Store" or tmp_cid == ".ipynb_checkpoints"):
            continue

        cid = str(tmp_cid)
        model_save_dir = os.path.join(log_dir, "models", desc, cid)
        if (not os.path.isdir(model_save_dir)):
            continue

        print("cid: ", cid)

        # -----------------------------------------
        # st_md_num = 800
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
            dream_pred_dir = os.path.join(dream_pred_par_dir, cid, md_fn)
            os.makedirs(dream_pred_dir, exist_ok=True)

            dream_infer_multi(model_dir=cur_model_dir, dream_pred_dir=dream_pred_dir,
                              caption_list_train=caption_list_train, pretrained_model_name=cfg.pretrained_model_name_or_path, lora_scale_unet=cfg.lora_scale_unet, lora_scale_text_encoder=cfg.lora_scale_text_encoder, instance_prompt=cfg.instance_prompt, scheduler_name=cfg.scheduler_name, base_steps=cfg.base_steps, base_scale=cfg.base_scale, num_prompt_samp=cfg.num_prompt_samp, num_samples=cfg.num_samples, num_rows=cfg.num_rows)
