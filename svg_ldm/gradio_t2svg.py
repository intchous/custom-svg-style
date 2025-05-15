import argparse
import time
import os
import argparse
import numpy as np
import random
import yaml
import cairosvg
import PIL
import uuid
import shutil

import torch
from safetensors.torch import load_file

from svg_ldm.config import _DefaultConfig

from svg_ldm.test_tr_utils import test_vecdiff_random_gen_prompts_pts
from svg_ldm.train_tr_utils import get_diffusion_models

import gradio as gr


def get_vecdiff_info():

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
    max_paths_len_thresh = cfg.max_paths_len_thresh
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

    return model_ddpm, cfg, sd_text_encoder, sd_tokenizer, save_svg_dir, max_paths_len_thresh, samp_method, guidance_scale, h, w
    # ---------------------------------------


model_ddpm, cfg, sd_text_encoder, sd_tokenizer, save_svg_dir, max_paths_len_thresh, samp_method, guidance_scale, h, w = get_vecdiff_info()


def predict(prompt, test_num=1, ddim_num_steps=100, samp_method="ddpm", guidance_scale=1.0):
    # generator = torch.manual_seed(seed)
    last_time = time.time()

    model_ddpm.make_ddim_schedule(ddim_num_steps=ddim_num_steps)

    # ---------------------------------------
    prompts_list = [prompt]

    # 给定prompts list进行测试(作为一个batch), 进行test_num次
    test_vecdiff_random_gen_prompts_pts(cfg=cfg, model_ddpm=model_ddpm, sd_text_encoder=sd_text_encoder, sd_tokenizer=sd_tokenizer, save_cubic_svg_dir=save_svg_dir, prompts_list=prompts_list,
                                        prompt_appd="", max_paths_len_thresh=max_paths_len_thresh, test_num=test_num, samp_method=samp_method, guidance_scale=guidance_scale, do_random_captions=False, canvas_width=w, canvas_height=h)

    png_paths, svg_paths = [], []
    fn_pre = prompt.replace("/", "_")
    for idx in range(test_num):
        orig_svg = os.path.join(save_svg_dir, f"rand_{fn_pre}_{idx}.svg")

        uid = uuid.uuid4().hex
        # tmp_svg = os.path.join(tempfile.gettempdir(), f"{uid}.svg")
        tmp_svg = os.path.join(save_svg_dir, f"{uid}.svg")
        tmp_png = tmp_svg.replace(".svg", ".png")
        shutil.copy(orig_svg, tmp_svg)

        cairosvg.svg2png(url=tmp_svg, write_to=tmp_png,
                         output_width=w, output_height=h)

        png_paths.append(tmp_png)
        svg_paths.append(tmp_svg)

    print(f"Inference took {time.time() - last_time:.2f}s, | n={test_num}")

    return png_paths, svg_paths


with gr.Blocks(css="""
    .container {max-width: 1000px; margin: auto;}
    .gr-gallery img {object-fit: cover; aspect-ratio: 1/1;}
""") as demo:

    gr.Markdown("# Text2SVG Generation", elem_classes="container")

    with gr.Row(elem_classes="container"):
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt", placeholder="Enter your prompt")
            generate_btn = gr.Button("Generate", variant="primary")

            with gr.Accordion("Advanced Options", open=True):
                # seed = gr.Slider(label="Seed", minimum=0,
                #                  maximum=9999999, step=1, randomize=True)
                test_num = gr.Slider(
                    label="Generation Count", minimum=1, maximum=10, step=1, value=1)
                ddim_num_steps = gr.Slider(
                    label="DDIM Steps", minimum=10, maximum=200, step=1, value=100)
                samp_method = gr.Radio(label="Sampling Method", choices=[
                                       "ddpm", "ddim"], value="ddpm")
                guidance_scale = gr.Slider(
                    label="Guidance Scale", minimum=1.0, maximum=12.0, step=0.5, value=4.0)

        with gr.Column(scale=1):
            result_gallery = gr.Gallery(
                label="Generated Images", show_label=True, columns=[3], height="auto")
            result_svg_files = gr.Files(label="Download SVG Files")

    generate_btn.click(
        predict,
        inputs=[prompt, test_num, ddim_num_steps, samp_method, guidance_scale],
        outputs=[result_gallery, result_svg_files],
        show_progress=True,
    )


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=0 python -m svg_ldm.gradio_t2svg
    demo.launch(server_name="0.0.0.0", show_api=False)
