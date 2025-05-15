import os
import argparse
import numpy as np
import random
import yaml

import torch
from safetensors.torch import load_file

from svg_ldm.config import _DefaultConfig
from svg_ldm.test_tr_utils import test_vecdiff_random_gen_prompts_pts
from svg_ldm.train_tr_utils import get_diffusion_models


if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=0 python -m svg_ldm.test_ddpm_tr_svgs

    parser = argparse.ArgumentParser(description="Load config from yaml file")
    parser.add_argument("--yaml_fn", type=str, default="test_ddpmacc_tr_svgs",
                        help="Path name of the yaml config file")
    args = parser.parse_args()

    cfg = _DefaultConfig()

    yaml_fp = os.path.join("./config_files/", args.yaml_fn + ".yaml")
    with open(yaml_fp, 'r') as f:
        config_data = yaml.safe_load(f)

    for key, value in config_data.items():
        setattr(cfg, key, value)

    # ---------------------------------------
    input_dim = cfg.n_args
    output_dim = cfg.n_args
    max_paths_len_thresh = cfg.max_paths_len_thresh
    max_points = cfg.max_points

    # batch_size = cfg.batch_size
    h, w = 224, 224

    # 50, 100
    ddim_num_steps = cfg.ddim_num_steps
    # ddpm, ddim
    samp_method = cfg.samp_method
    guidance_scale = cfg.guidance_scale
    # ------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ddpm, sd_text_encoder, sd_tokenizer = get_diffusion_models(
        cfg=cfg, model_type="ddpm", accelerator=None, device=device)

    # ---------------------------------------
    img_cid_list = ["1944", "6856", "6362"]
    log_dir = "./ldm_vae_logs/"
    save_svg_dir = os.path.join(log_dir, "test", cfg.pretrained_fn)
    os.makedirs(save_svg_dir, exist_ok=True)

    ddpm_model_fp = os.path.join(
        "./pretrained", cfg.pretrained_fn, "model.safetensors")
    state_dict = load_file(ddpm_model_fp)
    model_ddpm_md = model_ddpm.model
    model_ddpm_md.load_state_dict(state_dict)
    model_ddpm = model_ddpm.to(device)

    model_ddpm.make_ddim_schedule(ddim_num_steps=ddim_num_steps)

    samp_prompts_times = 1
    test_num = 4

    prompts_list = ["love window, sign", "twinkle",
                    "month calendar, calendar", "weighing scale"]

    # ---------------------------------------
    for _ in range(samp_prompts_times):
        if (cfg.add_style_token):
            style_cid = random.choice(img_cid_list)
            if (style_cid == "1944" or style_cid == "19252"):
                prompts_list = [
                    f"{prompt}; iconfont_{style_cid} style" for prompt in prompts_list]
            else:
                prompts_list = [
                    f"{prompt}; svgrepo_{style_cid} style" for prompt in prompts_list]

        test_vecdiff_random_gen_prompts_pts(cfg=cfg, model_ddpm=model_ddpm, sd_text_encoder=sd_text_encoder, sd_tokenizer=sd_tokenizer, save_cubic_svg_dir=save_svg_dir, prompts_list=prompts_list,
                                            prompt_appd="", max_paths_len_thresh=max_paths_len_thresh, test_num=test_num, samp_method=samp_method, guidance_scale=guidance_scale, do_random_captions=False, canvas_width=w, canvas_height=h)
    # ---------------------------------------
