import random
import tqdm
import os
import torch

from svg_ldm.Vecdiffusion.model_diffusion_ddpm import encode_prompt

from svg_ldm.train_tr_utils import cont_pidx_to_binary, pts_to_pathObj, random_captions, random_captions_style
from svg_ldm.svg_dataset_ldm import binary_tensor_to_decimal

import pydiffvg


def truncate_filename(original_path, max_length=250):
    directory, filename = os.path.split(original_path)
    name, ext = os.path.splitext(filename)

    # 计算允许的文件名长度（考虑扩展名）
    max_name_length = max_length - len(ext)

    if len(name) > max_name_length:
        truncated_name = name[:max_name_length]
        new_filename = truncated_name + ext
    else:
        new_filename = filename

    new_fp = os.path.join(directory, new_filename)

    return new_fp


def get_text_emb(captions, sd_text_encoder, sd_tokenizer, use_glb=False, do_classifier_free_guidance=False):

    text_embeddings = None
    encoder_hidden_states_uncond = None
    encoder_hidden_states_cond = None

    with torch.no_grad():
        text_embeddings = encode_prompt(
            prompt=captions, text_encoder=sd_text_encoder, tokenizer=sd_tokenizer, do_classifier_free_guidance=False, use_glb=use_glb)

    text_embeddings = text_embeddings.to(torch.float32)

    if (use_glb):
        text_embeddings = text_embeddings.unsqueeze(1)

    if (do_classifier_free_guidance):
        encoder_hidden_states_cond = text_embeddings
        encoder_hidden_states_uncond = torch.zeros_like(
            encoder_hidden_states_cond)
        text_embeddings = torch.cat(
            [encoder_hidden_states_uncond, encoder_hidden_states_cond])

    return text_embeddings


# --------------------------------------------
def svg_diffusion_infer(model, cfg, encoder_hidden_states=None, samp_method="ddpm", batch_size=1, max_len=25, seq_dim=32, guidance_scale=1.0):

    model.eval()

    paths_intermediates = None
    y_inters = None
    pred_y0s = None

    with torch.no_grad():
        if (samp_method == "ddim"):
            if (cfg.use_LCScheduler):
                paths_t_0 = model.reverse_ddim_lc(
                    batch_size=batch_size, encoder_hidden_states=encoder_hidden_states, stochastic=True, max_len=max_len, seq_dim=seq_dim, guidance_scale=guidance_scale)
            else:
                paths_t_0 = model.reverse_ddim_scheduler(
                    batch_size=batch_size, encoder_hidden_states=encoder_hidden_states, stochastic=True, max_len=max_len, seq_dim=seq_dim, guidance_scale=guidance_scale)
        else:
            if (cfg.use_LCScheduler):
                paths_t_0 = model.reverse_ddpm_lc(
                    batch_size=batch_size, encoder_hidden_states=encoder_hidden_states, stochastic=True, max_len=max_len, seq_dim=seq_dim, guidance_scale=guidance_scale)
            else:
                paths_t_0 = model.reverse_ddpm_scheduler(
                    batch_size=batch_size, encoder_hidden_states=encoder_hidden_states, stochastic=True, max_len=max_len, seq_dim=seq_dim, guidance_scale=guidance_scale)

    return paths_t_0, paths_intermediates, y_inters, pred_y0s


# --------------------------------------------
def load_svg_from_restensor_test(pts_feat_tensor, path_idx_tensor, max_paths_len_thresh=31, only_black_white_color=False, canvas_width=224, canvas_height=224):

    device = pts_feat_tensor.device
    norm_tensor = torch.tensor(
        [canvas_width, canvas_height], dtype=torch.float32, device=device)

    if (only_black_white_color):
        num_bits = pts_feat_tensor.shape[1] - 3
    else:
        num_bits = pts_feat_tensor.shape[1] - 6

    tp_shapes = []
    tp_shape_groups = []
    current_path_points = []
    current_path_idx = None
    real_current_path_idx = 0

    for _i, features in enumerate(pts_feat_tensor):
        path_idx = path_idx_tensor[_i]
        path_idx = path_idx.clamp(min=0)

        if (path_idx.item() == max_paths_len_thresh):
            break

        x = features[num_bits] * norm_tensor[0]
        x = torch.clamp(x, 0.001, norm_tensor[0]-0.001)

        y = features[num_bits + 1] * norm_tensor[1]
        y = torch.clamp(y, 0.001, norm_tensor[1]-0.001)

        if (only_black_white_color):
            color = features[num_bits + 2:num_bits + 3]
            color = torch.cat(
                [color, color, color, torch.tensor([1.0], device=device)])
        else:
            color = features[num_bits + 2:num_bits + 6]

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

                current_path_points = []

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


def recon_data_to_svgs_ldm_test(recon_data_output, tmp_svg_path_fp_list=[], max_paths_len_thresh=31, only_black_white_color=False, canvas_width=224, canvas_height=224):
    if (only_black_white_color):
        num_bits = recon_data_output.shape[-1] - 3
    else:
        num_bits = recon_data_output.shape[-1] - 6

    tmp_path_idx_tensors = recon_data_output[:, :, :num_bits]
    if (num_bits == 1):
        max_path_idx = max_paths_len_thresh
        path_idx_tensors = cont_pidx_to_binary(
            tmp_path_idx_tensors, max_path_idx=max_path_idx)
    else:
        # [B, max_points]
        path_idx_tensors = binary_tensor_to_decimal(tmp_path_idx_tensors)

    # print("path_idx_tensors:\n", path_idx_tensors)

    tp_shapes_list = []
    tp_shape_groups_list = []
    for b_idx in range(0, recon_data_output.shape[0]):
        cur_pts_feat_tensor = recon_data_output[b_idx]
        cur_path_idx_tensor = path_idx_tensors[b_idx]

        tp_shapes, tp_shape_groups = load_svg_from_restensor_test(
            pts_feat_tensor=cur_pts_feat_tensor, path_idx_tensor=cur_path_idx_tensor, max_paths_len_thresh=max_paths_len_thresh, only_black_white_color=only_black_white_color, canvas_width=canvas_width, canvas_height=canvas_height)

        if (len(tmp_svg_path_fp_list) > b_idx):
            tmp_svg_path_fp = tmp_svg_path_fp_list[b_idx]
            tmp_svg_path_fp = truncate_filename(tmp_svg_path_fp)
            pydiffvg.save_svg(tmp_svg_path_fp, canvas_width,
                              canvas_height, tp_shapes, tp_shape_groups)

        tp_shapes_list.append(tp_shapes)
        tp_shape_groups_list.append(tp_shape_groups)

    return tp_shapes_list, tp_shape_groups_list


# --------------------------------------------
def test_gen_svgs(model, cfg, encoder_hidden_states=None, samp_method="ddpm", batch_size=1, max_paths_len_thresh=31, guidance_scale=1.0, tmp_svg_path_fp_list=[],  canvas_width=224, canvas_height=224):

    paths_t_0_1, _, _, _ = svg_diffusion_infer(
        model=model, cfg=cfg, encoder_hidden_states=encoder_hidden_states, samp_method=samp_method, batch_size=batch_size, max_len=cfg.max_total_len, seq_dim=cfg.n_args, guidance_scale=guidance_scale)

    # [-1, 1] -> [0, 1]
    rand_gen_latents = (paths_t_0_1 + 1.0) * 0.5
    # [B, max_points, num_bits]

    recon_data_to_svgs_ldm_test(recon_data_output=rand_gen_latents, tmp_svg_path_fp_list=tmp_svg_path_fp_list, max_paths_len_thresh=max_paths_len_thresh,
                                only_black_white_color=cfg.only_black_white_color, canvas_width=canvas_width, canvas_height=canvas_height)


# ----------------------------------------------
# 给定train_loader, 取第一个batch(batch_size个)prompts进行测试, 进行test_num次
def test_vecdiff_random_gen_trl_pts(cfg, model_ddpm, train_loader, sd_text_encoder, sd_tokenizer, save_cubic_svg_dir, prompt_appd="", max_paths_len_thresh=31, batch_size=1, test_num=4, samp_method="ddpm", guidance_scale=1.0, do_random_captions=False, canvas_width=224, canvas_height=224):
    do_classifier_free_guidance = guidance_scale > 1.0

    batch_data = next(iter(train_loader))
    captions = batch_data["caption"]

    if (do_random_captions):
        if (cfg.add_style_token):
            pre_caption = random_captions_style(captions=captions, max_words=2)
        else:
            pre_caption = random_captions(captions=captions, max_words=2)
    else:
        pre_caption = captions[:]

    if (len(prompt_appd) > 0):
        pre_caption_sty = [prompt_appd + ', ' + c for c in pre_caption]
        pre_caption.extend(pre_caption_sty)

    batch_size = len(pre_caption)

    encoder_hidden_states = None
    if cfg.label_condition:
        print("caption: ", pre_caption)

        encoder_hidden_states = get_text_emb(
            captions=pre_caption,
            sd_text_encoder=sd_text_encoder,
            sd_tokenizer=sd_tokenizer,
            use_glb=cfg.use_glb,
            do_classifier_free_guidance=do_classifier_free_guidance
        )

    # test random generation
    for t_i in range(test_num):
        rand_samp_fp_list1 = [os.path.join(
            save_cubic_svg_dir, f"rand1_{c.replace('/', '_')}_{t_i}.svg") for c in pre_caption]

        # ---------------------------------------
        test_gen_svgs(model=model_ddpm, cfg=cfg, encoder_hidden_states=encoder_hidden_states, samp_method=samp_method, batch_size=batch_size,
                      max_paths_len_thresh=max_paths_len_thresh, guidance_scale=guidance_scale, tmp_svg_path_fp_list=rand_samp_fp_list1,  canvas_width=canvas_width, canvas_height=canvas_height)

    return


# ----------------------------------------------
# 给定空prompts list进行测试(作为一个batch), 进行test_num次
def test_vecdiff_uncond_gen_pts(cfg, model_ddpm, sd_text_encoder, sd_tokenizer, save_cubic_svg_dir, max_paths_len_thresh=31, batch_size=1, test_num=4, samp_method="ddpm", canvas_width=224, canvas_height=224):

    if not cfg.label_condition:
        return

    pre_caption = [""] * batch_size

    text_embeddings = get_text_emb(
        captions=pre_caption,
        sd_text_encoder=sd_text_encoder,
        sd_tokenizer=sd_tokenizer,
        use_glb=cfg.use_glb,
        do_classifier_free_guidance=False
    )
    encoder_hidden_states = torch.zeros_like(text_embeddings)

    for m_i in range(test_num):
        rand_samp_fp_list = [os.path.join(
            save_cubic_svg_dir, f"rand_uncond_{m_i}_{ba_i}.svg") for ba_i in range(len(pre_caption))]

        # ---------------------------------------
        test_gen_svgs(model=model_ddpm, cfg=cfg, encoder_hidden_states=encoder_hidden_states, samp_method=samp_method, batch_size=batch_size,
                      max_paths_len_thresh=max_paths_len_thresh, guidance_scale=1.0, tmp_svg_path_fp_list=rand_samp_fp_list, canvas_width=canvas_width, canvas_height=canvas_height)

    return


# ----------------------------------------------
# 给定prompts list进行测试(作为一个batch), 进行test_num次
def test_vecdiff_random_gen_prompts_pts(cfg, model_ddpm, sd_text_encoder, sd_tokenizer, save_cubic_svg_dir, prompts_list=[], prompt_appd="", max_paths_len_thresh=31, test_num=4, samp_method="ddpm", guidance_scale=1.0, do_random_captions=True, canvas_width=224, canvas_height=224):

    if not cfg.label_condition or not prompts_list:
        return

    pre_caption = [p.replace("/", ", ") for p in prompts_list]

    if (do_random_captions):
        if (cfg.add_style_token):
            pre_caption = random_captions_style(
                captions=pre_caption, max_words=2)
        else:
            pre_caption = random_captions(captions=pre_caption, max_words=2)
    # ------------------------------

    if (len(prompt_appd) > 0):
        pre_caption_sty = [f"{prompt_appd}, {c}" for c in pre_caption]
        pre_caption.extend(pre_caption_sty)

    do_classifier_free_guidance = guidance_scale > 1.0

    encoder_hidden_states = None
    if cfg.label_condition:
        print("caption: ", pre_caption)

        encoder_hidden_states = get_text_emb(
            captions=pre_caption,
            sd_text_encoder=sd_text_encoder,
            sd_tokenizer=sd_tokenizer,
            use_glb=cfg.use_glb,
            do_classifier_free_guidance=do_classifier_free_guidance
        )

    for m_i in tqdm.tqdm(range(test_num),
                         desc=f"Generating {test_num} times",
                         unit="batch",):

        rand_samp_fp_list = [os.path.join(save_cubic_svg_dir, f"rand_{c.replace('/', '_')}_{m_i}.svg")
                             for c in pre_caption]

        # ---------------------------------------
        test_gen_svgs(model=model_ddpm, cfg=cfg, encoder_hidden_states=encoder_hidden_states, samp_method=samp_method, batch_size=len(
            pre_caption), max_paths_len_thresh=max_paths_len_thresh, guidance_scale=guidance_scale, tmp_svg_path_fp_list=rand_samp_fp_list,  canvas_width=canvas_width, canvas_height=canvas_height)

    return


# ----------------------------------------------
# 给定train_loader, 在里面随机选取batch_size个data进行reconstruction, 并用prompts进行随机生成, 进行test_num次
def test_vecdiff_rec_gen_trl_pts(cfg, model_ddpm, train_loader, sd_text_encoder, sd_tokenizer, save_cubic_svg_dir, prompt_appd="", max_paths_len_thresh=31, batch_size=1, test_num=4, samp_method="ddpm", guidance_scale=1.0, do_random_captions=False, rec_sample_t_max=100, canvas_width=224, canvas_height=224):

    do_classifier_free_guidance = guidance_scale > 1.0

    batch_data = next(iter(train_loader))
    captions = batch_data["caption"]

    if (do_random_captions):
        if (cfg.add_style_token):
            pre_caption = random_captions_style(captions=captions, max_words=2)
        else:
            pre_caption = random_captions(captions=captions, max_words=2)
    else:
        pre_caption = captions[:]

    if (len(prompt_appd) > 0):
        pre_caption_sty = [prompt_appd + ', ' + c for c in pre_caption]
        pre_caption.extend(pre_caption_sty)

    batch_size = len(pre_caption)

    pre_caption_encoder_hidden_states = None
    if cfg.label_condition:
        print("caption: ", pre_caption)

        pre_caption_encoder_hidden_states = get_text_emb(
            captions=pre_caption,
            sd_text_encoder=sd_text_encoder,
            sd_tokenizer=sd_tokenizer,
            use_glb=cfg.use_glb,
            do_classifier_free_guidance=do_classifier_free_guidance
        )

    for t_i, batch_data in enumerate(train_loader):
        if t_i > test_num:
            break

        points_batch_real = batch_data["padded_pts_tensor"]
        points_batch_real = points_batch_real.to(model_ddpm.device)

        filepaths = batch_data["filepaths"]
        captions = batch_data["caption"]

        ini_svg_fp_list = []
        rec_svg_fp_list = []
        rand_samp_fp_list = []
        for ba_i in range(points_batch_real.shape[0]):
            ini_svg_fp = filepaths[ba_i]
            ini_svg_fp_pre = ini_svg_fp.split("/")[-1].split(".")[0]
            ini_svg_fp_pre = ini_svg_fp_pre.replace("_rsz_feat", "")

            tmp_svg_path_fp = os.path.join(
                save_cubic_svg_dir, ini_svg_fp_pre + "_ini.svg")
            ini_svg_fp_list.append(tmp_svg_path_fp)

            tmp_svg_path_fp = os.path.join(
                save_cubic_svg_dir, ini_svg_fp_pre + "_recons.svg")
            rec_svg_fp_list.append(tmp_svg_path_fp)

        for ba_i in range(len(pre_caption)):
            tmp_caption = pre_caption[ba_i]
            fn_pre = tmp_caption.replace("/", "_")

            tmp_svg_path_fp = os.path.join(
                save_cubic_svg_dir, "rand_" + str(fn_pre) + "_" + str(t_i) + ".svg")
            rand_samp_fp_list.append(tmp_svg_path_fp)

        with torch.no_grad():
            data_pts = points_batch_real

            # [0, 1] -> [-1, 1]
            data_pts = data_pts * 2.0 - 1.0

            # ---------------------------------------
            text_embeddings = None
            if (cfg.label_condition):
                print("captions: ", captions)

                text_embeddings = get_text_emb(captions=captions, sd_text_encoder=sd_text_encoder,
                                               sd_tokenizer=sd_tokenizer, use_glb=cfg.use_glb, do_classifier_free_guidance=False)

            encoder_hidden_states = text_embeddings
            # ---------------------------------------

            discrete_t = model_ddpm.sample_t(
                [data_pts.shape[0]], t_max=rec_sample_t_max)

            if (cfg.use_LCScheduler):
                eps_theta, e, b_0_reparam, weighing = model_ddpm.forward_t(
                    l_0_batch=data_pts, t=discrete_t, encoder_hidden_states=encoder_hidden_states, reparam=True)
            else:
                eps_theta, e, b_0_reparam, weighing = model_ddpm.forward_t_ddpm_scheduler(
                    l_0_batch=data_pts, t=discrete_t, encoder_hidden_states=encoder_hidden_states)
            # ------------------------------

        test_gen_svgs(model=model_ddpm, cfg=cfg, encoder_hidden_states=pre_caption_encoder_hidden_states, samp_method=samp_method, batch_size=batch_size,
                      max_paths_len_thresh=max_paths_len_thresh, guidance_scale=guidance_scale, tmp_svg_path_fp_list=rand_samp_fp_list,  canvas_width=canvas_width, canvas_height=canvas_height)

        # [-1, 1] -> [0, 1]
        rec_gen_latents = (b_0_reparam + 1.0) * 0.5

        real_data_output = points_batch_real
        recon_data_output = rec_gen_latents

        # ---------------------------------------------
        recon_data_to_svgs_ldm_test(recon_data_output=real_data_output, tmp_svg_path_fp_list=ini_svg_fp_list,
                                    max_paths_len_thresh=max_paths_len_thresh, only_black_white_color=cfg.only_black_white_color, canvas_width=canvas_width, canvas_height=canvas_height)

        recon_data_to_svgs_ldm_test(recon_data_output=recon_data_output, tmp_svg_path_fp_list=rec_svg_fp_list, max_paths_len_thresh=max_paths_len_thresh,
                                    only_black_white_color=cfg.only_black_white_color, canvas_width=canvas_width, canvas_height=canvas_height)

        # for tmp_img_render in tmp_img_render_list:
        #     tmp_img_path_fp = os.path.join(
        #         save_cubic_svg_dir, "randz_gen_" + str(i) + ".png")
        #     # 注意要有足够的cuda内存, 不然渲染出来的图片是空白
        #     pydiffvg.imwrite(tmp_img_render.cpu(),
        #                      tmp_img_path_fp, gamma=1.0)
        #     # vutils.save_image(recon_imgs, tmp_img_path_fp, nrow=1, padding=1, normalize=False)

    return


# ----------------------------------------------
def test_vecdiff_pts(cfg, model_ddpm, train_loader, sd_text_encoder, sd_tokenizer, save_cubic_svg_dir, prompts_list=[], prompt_appd="", max_paths_len_thresh=31, batch_size=1, test_num=4, samp_method="ddpm", guidance_scale=1.0, rec_sample_t_max=100, canvas_width=224, canvas_height=224):

    test_vecdiff_random_gen_trl_pts(cfg=cfg, model_ddpm=model_ddpm, train_loader=train_loader, sd_text_encoder=sd_text_encoder, sd_tokenizer=sd_tokenizer, save_cubic_svg_dir=save_cubic_svg_dir, prompt_appd=prompt_appd,
                                    max_paths_len_thresh=max_paths_len_thresh, batch_size=batch_size, test_num=test_num, samp_method=samp_method, guidance_scale=guidance_scale, do_random_captions=True, canvas_width=canvas_width, canvas_height=canvas_height)

    # ---------------------------------------
    uncond_batch_size = 2
    uncond_test_num = 1
    test_vecdiff_uncond_gen_pts(cfg=cfg, model_ddpm=model_ddpm, sd_text_encoder=sd_text_encoder, sd_tokenizer=sd_tokenizer, save_cubic_svg_dir=save_cubic_svg_dir,
                                max_paths_len_thresh=max_paths_len_thresh, batch_size=uncond_batch_size, test_num=uncond_test_num, samp_method=samp_method, canvas_width=canvas_width, canvas_height=canvas_height)

    # ---------------------------------------
    test_vecdiff_random_gen_prompts_pts(cfg=cfg, model_ddpm=model_ddpm, sd_text_encoder=sd_text_encoder, sd_tokenizer=sd_tokenizer, save_cubic_svg_dir=save_cubic_svg_dir, prompts_list=prompts_list,
                                        prompt_appd=prompt_appd, max_paths_len_thresh=max_paths_len_thresh, test_num=test_num, samp_method=samp_method, guidance_scale=guidance_scale, do_random_captions=True, canvas_width=canvas_width, canvas_height=canvas_height)

    # ---------------------------------------
    # train_loader 由于已经确定了batch_size, 所以这里recon_batch_size不能改
    recon_batch_size = batch_size
    recon_test_num = 1
    test_vecdiff_rec_gen_trl_pts(cfg=cfg, model_ddpm=model_ddpm, train_loader=train_loader, sd_text_encoder=sd_text_encoder, sd_tokenizer=sd_tokenizer, save_cubic_svg_dir=save_cubic_svg_dir, prompt_appd=prompt_appd,
                                 max_paths_len_thresh=max_paths_len_thresh, batch_size=recon_batch_size, test_num=recon_test_num, samp_method=samp_method, guidance_scale=guidance_scale, do_random_captions=True, rec_sample_t_max=rec_sample_t_max, canvas_width=canvas_width, canvas_height=canvas_height)
    # --------------------------------------------
