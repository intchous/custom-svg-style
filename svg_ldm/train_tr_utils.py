import random
import re
import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer

from svg_ldm.Vecdiffusion.model_diffusion_ddpm import Diffusion_ddpm, encode_prompt

import pydiffvg


def log_and_write(epoch, loss, batch_idx=-1, len_dataloader=100, loss_type="train", summary_writer=None):
    # Check if loss is a tensor and retrieve its value, else use it directly
    loss_val = loss.item() if torch.is_tensor(loss) else loss

    if batch_idx != -1:
        print(
            f'Epoch {epoch} [{batch_idx}/{len_dataloader}] {loss_type} Loss: {loss_val}')
        if summary_writer:
            step = epoch * len_dataloader + batch_idx
            summary_writer.add_scalar(f'Loss/{loss_type}', loss_val, step)

    else:
        print(f'Epoch {epoch} {loss_type} Loss: {loss_val}')
        if summary_writer:
            summary_writer.add_scalar(
                f'Loss/{loss_type}_epoch_mean', loss_val, epoch)


def get_desc_pts(cfg, signature="IconShop_diffvg_select", desc_prefix="ddpmacc_tr_fsvgs_pts_"):
    desc = desc_prefix + "dt-" + signature + "_nl-" + str(cfg.nlayer) + "_trdim-" + str(cfg.dim_transformer) + "_indim-" + str(cfg.feature_dim) + "_nh-" + str(cfg.nhead) + "_losstype-" + str(cfg.loss_type) + "_txtglb-" + str(
        cfg.use_glb) + "_lbcond-" + str(cfg.label_condition) + "_simth-" + str(cfg.similarity_threshold) + "_cluth-" + str(cfg.select_threshold) + "_mpaths-" + str(cfg.max_paths_len_thresh) + "_mpts-" + str(cfg.max_points)

    if (cfg.only_black_white_color):
        desc = desc + "_bw"

    if (cfg.use_cont_path_idx):
        desc = desc + "_contpi"

    if (cfg.use_LCScheduler):
        desc = desc + "_LCS"

    if (cfg.discrete_t):
        assert cfg.use_norm_out == False
        desc = desc + "_deT"

    if (cfg.use_norm_out):
        desc = desc + "_nmout"

    if (cfg.add_style_token):
        desc = desc + "_stytk"

    return desc


# --------------------------------------------
def get_diffusion_models(cfg, model_type="ddpm", train_text_encoder=False, accelerator=None, device="cuda"):
    # set up model
    # diffusers, LACE

    sd_text_encoder = None
    sd_tokenizer = None

    if (cfg.label_condition):
        # encoder_hid_dim=768
        # cross_attention_dim = min(768, cfg.dim_transformer)
        encoder_hid_dim = None
        cross_attention_dim = 768

        if (model_type == "ddpm"):
            if (accelerator is None):
                print("Set Diffusion_ddpm model with cross_attention...")
            else:
                if accelerator.is_main_process:
                    print(
                        "Set Diffusion_ddpm model with cross_attention...")

            model_ddpm = Diffusion_ddpm(num_timesteps=1000, nhead=cfg.nhead, feature_dim=cfg.feature_dim, dim_transformer=cfg.dim_transformer, num_layers=cfg.nlayer, max_len=cfg.max_total_len,
                                             seq_dim=cfg.n_args, device=device, ddim_num_steps=200, encoder_hid_dim=encoder_hid_dim, cross_attention_dim=cross_attention_dim, use_norm_out=cfg.use_norm_out)

        pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

        sd_tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer")
        sd_text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder").to(device)

        if (train_text_encoder):
            sd_text_encoder.requires_grad_(True)
        else:
            sd_text_encoder.requires_grad_(False)

    else:
        if (model_type == "ddpm"):
            if (accelerator is None):
                print("Set Diffusion_ddpm model w/o cross_attention...")
            else:
                if accelerator.is_main_process:
                    print(
                        "Set Diffusion_ddpm model w/o cross_attention...")

            model_ddpm = Diffusion_ddpm(num_timesteps=1000, nhead=cfg.nhead, feature_dim=cfg.feature_dim, dim_transformer=cfg.dim_transformer, num_layers=cfg.nlayer,
                                             max_len=cfg.max_total_len, seq_dim=cfg.n_args, device=device, ddim_num_steps=200, encoder_hid_dim=None, cross_attention_dim=None, use_norm_out=cfg.use_norm_out)

    return model_ddpm, sd_text_encoder, sd_tokenizer


# --------------------------------------------
def get_encoder_hidden_states(cfg, captions, sd_text_encoder, sd_tokenizer, no_grad=True):
    text_embeddings = None

    if (cfg.label_condition):
        # captions = [c + ', ' + prompt_appd for c in captions]
        # negative_prompts = [""] * len(captions)

        if (no_grad):
            with torch.no_grad():
                text_embeddings = encode_prompt(prompt=captions, text_encoder=sd_text_encoder,
                                                tokenizer=sd_tokenizer, do_classifier_free_guidance=False, use_glb=cfg.use_glb)
        else:
            text_embeddings = encode_prompt(prompt=captions, text_encoder=sd_text_encoder,
                                            tokenizer=sd_tokenizer, do_classifier_free_guidance=False, use_glb=cfg.use_glb)

        text_embeddings = text_embeddings.to(torch.float32)
        # text_embeddings.shape:  torch.Size([batch_size, 77, 768])
        # text_embeddings.shape:  torch.Size([batch_size, 768]) -> torch.Size([batch_size, 1, 768])

        if (cfg.use_glb):
            text_embeddings = text_embeddings.unsqueeze(1)

    encoder_hidden_states = text_embeddings

    return encoder_hidden_states


# --------------------------------------------
def condition_dropout(conditioning_dropout_prob, encoder_hidden_states, bsz, device):
    # Conditioning dropout to support classifier-free guidance during inference. For more details
    # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.

    random_p = torch.rand(bsz, device=device)

    # Sample masks for the edit prompts.
    # prompt_mask = random_p < 2 * cfg.conditioning_dropout_prob
    prompt_mask = random_p < conditioning_dropout_prob
    prompt_mask = prompt_mask.reshape(bsz, 1, 1)

    # Final text conditioning.
    null_conditioning = torch.zeros_like(encoder_hidden_states)
    # null_conditioning.shape:  torch.Size([150, 77, 768])

    encoder_hidden_states = torch.where(
        prompt_mask, null_conditioning, encoder_hidden_states)
    # encoder_hidden_states.shape:  torch.Size([150, 77, 768])

    return encoder_hidden_states


def random_captions(captions, max_words=100):
    # 随机选择部分label, 并打乱顺序
    # 统一分隔符，将 '/' 替换为 ', '
    # captions = [caption.replace("/", ", ") for caption in captions]

    new_captions = []
    for caption in captions:
        # words = caption.split(', ')
        words = [word.strip() for word in caption.split(', ') if word.strip()]

        num_words = len(words)
        max_k = min(max_words, num_words)

        # 随机选择 1 到 max_k 个单词
        k = random.randint(1, max_k) if max_k > 0 else 0
        if k == 0:
            new_captions.append('')
            continue

        # 随机选择 k 个单词（已随机排序）
        selected_words = random.sample(words, k)
        # 重新组合成新的 caption
        new_caption = ', '.join(selected_words)
        new_captions.append(new_caption)

    return new_captions


def random_captions_style(captions, max_words=100):
    new_captions = []
    for caption in captions:
        # 提取并移除 "; xxx style"
        match = re.search(r"(; .*? style)$", caption)
        if match:
            style_sign = match.group(1)
            caption_wo_style = caption.replace(style_sign, "").strip()
        else:
            style_sign = ""
            caption_wo_style = caption

        words = [word.strip()
                 for word in caption_wo_style.split(', ') if word.strip()]

        if 'iconshop style' in caption:
            num_words = len(words)
            max_k = min(max_words, num_words)

            # 随机选择 1 到 max_k 个单词
            k = random.randint(1, max_k) if max_k > 0 else 0
            if k == 0:
                new_captions.append('')
                continue

            # 随机选择 k 个单词（已随机排序）
            selected_words = random.sample(words, k)
            # 重新组合成新的 caption
            new_caption = ', '.join(selected_words) + style_sign

        else:
            # 从前两个单词里随机选一个
            if len(words) >= 2:
                new_caption = random.choice(words[:2]) + style_sign
            elif len(words) == 1:
                new_caption = words[0] + style_sign
            else:
                new_caption = ''

        new_captions.append(new_caption)

    return new_captions


def get_first_word(caption):
    words = [word.strip() for word in caption.split(', ') if word.strip()]
    new_caption = words[0] if words else ''

    return new_caption


def get_first_word_style(caption):

    # 提取并移除 "; xxx style"
    match = re.search(r"(; .*? style)$", caption)
    if match:
        style_sign = match.group(1)
        caption_wo_style = caption.replace(style_sign, "").strip()
    else:
        style_sign = ""
        caption_wo_style = caption

    words = [word.strip()
             for word in caption_wo_style.split(', ') if word.strip()]

    new_caption = (words[0] + style_sign) if words else ''

    return new_caption


# --------------------------------------------
def load_tr_model_weights(new_model, checkpoint_path, pre_num_layers=12, new_num_layers=12):
    # state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = load_file(checkpoint_path)

    pretrained_state_dict = state_dict
    new_state_dict = new_model.state_dict()
    model_dict = {}

    if 'layer_in.weight' in pretrained_state_dict:
        # [dim_transformer, pre_seq_dim]
        pre_weight = pretrained_state_dict['layer_in.weight']
        # [dim_transformer, new_seq_dim]
        new_weight = new_state_dict['layer_in.weight']
        pre_seq_dim = pre_weight.shape[1]
        new_seq_dim = new_weight.shape[1]
        min_seq_dim = min(pre_seq_dim, new_seq_dim)

        new_weight[:, :min_seq_dim] = pre_weight[:, :min_seq_dim]
        if new_seq_dim > pre_seq_dim:
            # 初始化新增部分
            nn.init.xavier_uniform_(new_weight[:, pre_seq_dim:])
        model_dict['layer_in.weight'] = new_weight

    if 'layer_out.weight' in pretrained_state_dict:
        # [pre_seq_dim, dim_transformer]
        pre_weight = pretrained_state_dict['layer_out.weight']
        # [new_seq_dim, dim_transformer]
        new_weight = new_state_dict['layer_out.weight']
        pre_seq_dim = pre_weight.shape[0]
        new_seq_dim = new_weight.shape[0]
        min_seq_dim = min(pre_seq_dim, new_seq_dim)

        new_weight[:min_seq_dim, :] = pre_weight[:min_seq_dim, :]
        if new_seq_dim > pre_seq_dim:
            nn.init.xavier_uniform_(new_weight[pre_seq_dim:, :])
        model_dict['layer_out.weight'] = new_weight

    min_num_layers = min(pre_num_layers, new_num_layers)
    for i in range(min_num_layers):
        for key in pretrained_state_dict.keys():
            if key.startswith(f'layers.{i}.') and key in new_state_dict and pretrained_state_dict[key].shape == new_state_dict[key].shape:
                model_dict[key] = pretrained_state_dict[key]

    for key in pretrained_state_dict.keys():
        if key not in model_dict and key in new_state_dict and pretrained_state_dict[key].shape == new_state_dict[key].shape:
            model_dict[key] = pretrained_state_dict[key]

    new_state_dict.update(model_dict)
    new_model.load_state_dict(new_state_dict)

    return new_model


# --------------------------------------------
def cont_pidx_to_binary(tensors, max_path_idx=31):
    decimal_values = (tensors.squeeze(-1) * max_path_idx).round().long()
    return decimal_values


def pts_to_pathObj(convert_points):
    # Number of control points per segment is 2. Hence, calculate the total number of segments.
    num_segments = int(convert_points.shape[0] / 3)
    num_segments = max(num_segments, 1)
    num_control_points = [2] * num_segments
    num_control_points = torch.LongTensor(num_control_points)

    # Create a path object
    path = pydiffvg.Path(
        num_control_points=num_control_points,
        points=convert_points,
        stroke_width=torch.tensor(0.0),
        is_closed=True
    )

    return path


def load_svg_from_restensor(pts_feat_tensor, max_paths_len_thresh=31, only_black_white_color=False, canvas_width=224, canvas_height=224):

    device = pts_feat_tensor.device
    norm_tensor = torch.tensor(
        [canvas_width, canvas_height], dtype=torch.float32, device=device)

    num_bits = 1
    tp_shapes = []
    tp_shape_groups = []
    current_path_points = []
    current_path_idx = None
    real_current_path_idx = 0

    for features in pts_feat_tensor:
        path_idx = (features[0] * max_paths_len_thresh).round()
        path_idx = path_idx.clamp(min=0)

        if (path_idx.item() == max_paths_len_thresh):
            break

        x = features[num_bits] * norm_tensor[0]
        x = torch.clamp(x, 0.001, norm_tensor[0]-0.001)

        y = features[num_bits + 1] * norm_tensor[1]
        y = torch.clamp(y, 0.001, norm_tensor[1]-0.001)

        if (only_black_white_color):
            color = features[num_bits + 2:num_bits + 3]
            # color1.shape:  torch.Size([1])
            color = torch.cat(
                [color, color, color, torch.tensor([1.0], device=device)])
        else:
            color = features[num_bits + 2:num_bits + 6]

        # clamp [0, 1]
        color = torch.clamp(color, 0.0, 1.0)

        if current_path_idx is None or path_idx.item() != current_path_idx:

            if current_path_points:  # 如果当前有点，保存路径
                current_path_points_tensor = torch.stack(
                    current_path_points, dim=0)

                if (current_path_points_tensor.shape[0] >= 3):
                    ini_path = pts_to_pathObj(current_path_points_tensor)
                    tp_shapes.append(ini_path)

                    tp_path_group = pydiffvg.ShapeGroup(shape_ids=torch.LongTensor(
                        [real_current_path_idx]), fill_color=current_color, use_even_odd_rule=False)
                    tp_shape_groups.append(tp_path_group)

                    real_current_path_idx += 1

                current_path_points = []  # 清空路径点

                if real_current_path_idx == max_paths_len_thresh:
                    break

            current_path_idx = path_idx.item()
            current_color = color

        current_point = torch.stack([x, y])
        current_path_points.append(current_point)

    if current_path_points:
        current_path_points_tensor = torch.stack(current_path_points, dim=0)

        if (current_path_points_tensor.shape[0] >= 3):
            ini_path = pts_to_pathObj(current_path_points_tensor)
            tp_shapes.append(ini_path)

            tp_shape_groups.append(pydiffvg.ShapeGroup(
                shape_ids=torch.LongTensor([real_current_path_idx]),
                fill_color=current_color,
                use_even_odd_rule=False
            ))

    return tp_shapes, tp_shape_groups


def recon_data_to_svgs_ldm(recon_data_output, max_paths_len_thresh=31, only_black_white_color=False, canvas_width=224, canvas_height=224):

    diffvg_render = pydiffvg.RenderFunction.apply
    device = recon_data_output.device
    para_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=device)

    recon_imgs_list = []
    render_imgs_list = []
    tp_shapes_list = []
    tp_shape_groups_list = []

    for b_idx in range(0, recon_data_output.shape[0]):

        cur_pts_feat_tensor = recon_data_output[b_idx]

        tp_shapes, tp_shape_groups = load_svg_from_restensor(
            pts_feat_tensor=cur_pts_feat_tensor, max_paths_len_thresh=max_paths_len_thresh, only_black_white_color=only_black_white_color, canvas_width=canvas_width, canvas_height=canvas_height)

        tp_shapes_list.append(tp_shapes)
        tp_shape_groups_list.append(tp_shape_groups)

        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, tp_shapes, tp_shape_groups)
        img = diffvg_render(canvas_width, canvas_height,
                            2, 2, 0, None, *scene_args)

        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + para_bg * (1 - img[:, :, 3:4])
        render_imgs_list.append(img)

        # x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        img_chw = img.permute(2, 0, 1)
        # [0, 1] -> [-1, 1]
        img_chw = img_chw * 2.0 - 1.0
        recon_imgs_list.append(img_chw)

    return tp_shapes_list, tp_shape_groups_list, recon_imgs_list, render_imgs_list
# --------------------------------------------
