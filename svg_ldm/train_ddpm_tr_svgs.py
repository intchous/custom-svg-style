import os
import argparse
import yaml
from datetime import datetime
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from accelerate import Accelerator

from tensorboardX import SummaryWriter

from svg_ldm.config import _DefaultConfig
from svg_ldm.ldm_dataset_utils import get_data_pts_list, get_select_lbids_dict, get_dataset_pts
from svg_ldm.train_tr_utils import log_and_write, get_desc_pts, get_diffusion_models, get_encoder_hidden_states, condition_dropout, load_tr_model_weights, random_captions, get_first_word, random_captions_style, get_first_word_style


if __name__ == "__main__":

    # CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --main_process_port 27501 -m svg_ldm.train_ddpm_tr_svgs

    parser = argparse.ArgumentParser(description="Load config from yaml file")
    parser.add_argument("--yaml_fn", type=str, default="train_ddpmacc_tr_svgs",
                        help="Path name of the yaml config file")
    args = parser.parse_args()

    cfg = _DefaultConfig()

    yaml_fp = os.path.join("./deepsvg/config_files/", args.yaml_fn + ".yaml")
    with open(yaml_fp, 'r') as f:
        config_data = yaml.safe_load(f)

    for key, value in config_data.items():
        setattr(cfg, key, value)

    # ---------------------------------------
    input_dim = cfg.n_args
    output_dim = cfg.n_args
    max_paths_len_thresh = cfg.max_paths_len_thresh
    max_points = cfg.max_points

    learning_rate = cfg.learning_rate

    batch_size = cfg.batch_size
    num_epochs = 3001
    h, w = 224, 224
    scaling_factor = 1.0

    if (cfg.use_LCScheduler):
        sample_t_max = 999
    else:
        sample_t_max = 1000

    validate_interval = int(cfg.validate_interval)
    log_interval = validate_interval

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
    )
    # ------------------------------------
    img_cid_list = ['1944', '6774', '5355', '6213', '6362', '6731',
                    '6820', '5903', '6856', '6972', '19252', '6700', '5558', '6993']

    svg_dataset_train = None
    svg_dataset_test = None

    # ----------------------------
    signature = cfg.signature
    if ("IconShop" in signature):
        obj_sign = "IconShop_diffvgimg_merge_resize"
        par_dir = "./dataset/IconShop/"

        select_lbids_dict = get_select_lbids_dict(cfg=cfg, par_dir=par_dir)

        if accelerator.is_main_process:
            print("select_threshold: ", str(cfg.select_threshold))
            print("len_threshold_info: ", len(select_lbids_dict))

        # ---------------------------------------
        svg_data_dir, tensor_file_list_train, tensor_file_list_test, caption_list_train, caption_list_test = get_data_pts_list(
            obj_sign=obj_sign, signature=signature, par_dir=par_dir, max_paths_len_thresh=max_paths_len_thresh, max_points=max_points, select_lbids_dict=select_lbids_dict, add_style_token=cfg.add_style_token)

        if accelerator.is_main_process:
            print("IconShop len(tensor_file_list_train): ",
                  len(tensor_file_list_train))
            print("IconShop len(tensor_file_list_test): ",
                  len(tensor_file_list_test))

        svg_dataset_train, svg_dataset_test = get_dataset_pts(cfg=cfg, tensor_file_list_train=tensor_file_list_train, tensor_file_list_test=tensor_file_list_test,
                                                              caption_list_train=caption_list_train, caption_list_test=caption_list_test, signature=signature, svg_data_dir=svg_data_dir)

    # ----------------------------
    svgrepo_signature = cfg.svgrepo_signature
    if ("svgrepo_collections" in svgrepo_signature):
        svg_data_sign = "svgrepo_collections"
        obj_sign = f"{svg_data_sign}_diffvgimg_merge_resize"
        par_dir = os.path.join("./dataset/", svg_data_sign)

        svg_data_dir, tensor_file_list_train, tensor_file_list_test, caption_list_train, caption_list_test = get_data_pts_list(
            obj_sign=obj_sign, signature=svgrepo_signature, par_dir=par_dir, max_paths_len_thresh=max_paths_len_thresh, max_points=max_points, select_lbids_dict=None, img_cid_list=img_cid_list, add_style_token=cfg.add_style_token)

        if accelerator.is_main_process:
            print("svgrepo len(tensor_file_list_train): ",
                  len(tensor_file_list_train))
            print("svgrepo len(tensor_file_list_test): ",
                  len(tensor_file_list_test))

        svgrepo_svg_dataset_train, svgrepo_svg_dataset_test = get_dataset_pts(cfg=cfg, tensor_file_list_train=tensor_file_list_train, tensor_file_list_test=tensor_file_list_test,
                                                                              caption_list_train=caption_list_train, caption_list_test=caption_list_test, signature=svgrepo_signature, svg_data_dir=svg_data_dir)

        if ((svg_dataset_train is not None) and (svg_dataset_test is not None)):
            svg_dataset_train = ConcatDataset(
                [svg_dataset_train, svgrepo_svg_dataset_train])
            svg_dataset_test = ConcatDataset(
                [svg_dataset_test, svgrepo_svg_dataset_test])
        else:
            svg_dataset_train = svgrepo_svg_dataset_train
            svg_dataset_test = svgrepo_svg_dataset_test

    # ----------------------------
    # persistent_workers=True
    train_loader = DataLoader(svg_dataset_train, batch_size=cfg.batch_size,
                              num_workers=cfg.loader_num_workers, shuffle=True)
    test_loader = DataLoader(svg_dataset_test, batch_size=cfg.batch_size,
                             num_workers=cfg.loader_num_workers, shuffle=False)

    # ---------------------------------------
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = accelerator.device

    if ("svgrepo_collections" in svgrepo_signature):
        desc = get_desc_pts(cfg, signature=svgrepo_signature,
                            desc_prefix="ddpmacc_tr_fsvgs_pts_")
    else:
        desc = get_desc_pts(cfg, signature=signature,
                            desc_prefix="ddpmacc_tr_fsvgs_pts_")

    if accelerator.is_main_process:
        print("desc: ", desc)

    log_dir = "./ldm_vae_logs/"
    current_time = datetime.now().strftime("%b%d_%H:%M")

    if accelerator.is_main_process:
        experiment_identifier = f"{desc}-{current_time}"
        summary_writer = SummaryWriter(os.path.join(
            log_dir, "tensorboard", experiment_identifier))

    model_save_dir = os.path.join(log_dir, "models", desc)
    os.makedirs(model_save_dir, exist_ok=True)

    # ---------------------------------------
    model_ddpm, sd_text_encoder, sd_tokenizer = get_diffusion_models(
        cfg=cfg, model_type="ddpm", accelerator=accelerator, device=device)

    # ---------------------------------------
    # optimizer
    model_ddpm_md = model_ddpm.model
    optimizer = optim.Adam(model_ddpm_md.parameters(), lr=learning_rate, weight_decay=0.0, betas=(
        0.9, 0.999), amsgrad=False, eps=1e-08, capturable=True)
    # ---------------------------------------

    start_epoch = 0
    if cfg.resume_from_checkpoint_notmatch:
        checkpoint_dir = os.path.join(model_save_dir, cfg.pretrained_fn)

        if os.path.exists(checkpoint_dir):
            start_epoch = int(cfg.pretrained_fn)
            checkpoint_path = os.path.join(checkpoint_dir, 'model.safetensors')

            model_ddpm_md = load_tr_model_weights(
                new_model=model_ddpm_md,
                checkpoint_path=checkpoint_path,
                pre_num_layers=cfg.pre_nlayer,
                new_num_layers=cfg.nlayer
            )

    # ---------------------------------------
    model_ddpm_md, optimizer, train_loader = accelerator.prepare(
        model_ddpm_md, optimizer, train_loader)

    # ---------------------------------------
    if cfg.resume_from_checkpoint_match:
        checkpoint_dir = os.path.join(model_save_dir, cfg.pretrained_fn)

        if os.path.exists(checkpoint_dir):
            start_epoch = int(cfg.pretrained_fn)
            accelerator.load_state(checkpoint_dir)

            optimizer.param_groups[0]['capturable'] = True
            # ----------------------------------------

    if (accelerator.is_main_process):
        print("start_epoch: ", start_epoch)
    # ---------------------------------------

    mse_loss = nn.MSELoss()
    best_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        model_ddpm_md.train()
        epoch_train_loss = 0.0

        if (epoch) % validate_interval == 0:
            if accelerator.is_main_process:
                tmp_model_save_dir = os.path.join(model_save_dir, str(epoch))
                os.makedirs(tmp_model_save_dir, exist_ok=True)
                accelerator.save_state(tmp_model_save_dir)
            # ----------------------------------------

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=200, disable=not accelerator.is_main_process) as pbar:

            for i, batch_data in pbar:

                with accelerator.accumulate(model_ddpm_md):
                    points_batch = batch_data["padded_pts_tensor"]
                    mask = batch_data["p_mask"]
                    captions = batch_data["caption"]

                    # ------------------------------
                    if (epoch % 2 == 0):
                        if (cfg.add_style_token):
                            captions = random_captions_style(
                                captions=captions, max_words=2)
                        else:
                            captions = random_captions(
                                captions=captions, max_words=2)
                    else:
                        if (cfg.add_style_token):
                            captions = [get_first_word_style(
                                caption) for caption in captions]
                        else:
                            captions = [get_first_word(
                                caption) for caption in captions]
                    # ------------------------------

                    data_pts = points_batch
                    # [0, 1] -> [-1, 1]
                    data_pts = data_pts * 2.0 - 1.0
                    data_pts = data_pts.detach()

                    # ------------------------------
                    encoder_hidden_states = get_encoder_hidden_states(
                        cfg=cfg, captions=captions, sd_text_encoder=sd_text_encoder, sd_tokenizer=sd_tokenizer, no_grad=True)

                    discrete_t = model_ddpm.sample_t(
                        [data_pts.shape[0]], t_max=sample_t_max)
                    # ------------------------------

                    # Conditioning dropout to support classifier-free guidance during inference. For more details
                    # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                    if (cfg.conditioning_dropout_prob > 0) and (encoder_hidden_states is not None):
                        encoder_hidden_states = condition_dropout(
                            conditioning_dropout_prob=cfg.conditioning_dropout_prob, encoder_hidden_states=encoder_hidden_states, bsz=points_batch.shape[0], device=device)

                    # ------------------------------
                    weighing = None
                    cal_x0 = ("x0rec" in cfg.loss_type)

                    if (cfg.use_LCScheduler):
                        eps_theta, e, b_0_reparam, weighing = model_ddpm.forward_t(
                            l_0_batch=data_pts, t=discrete_t, encoder_hidden_states=encoder_hidden_states, reparam=cal_x0)
                    else:
                        eps_theta, e, b_0_reparam, weighing = model_ddpm.forward_t_ddpm_scheduler(
                            l_0_batch=data_pts, t=discrete_t, encoder_hidden_states=encoder_hidden_states)
                    # ------------------------------
                    if ("x0rec" in cfg.loss_type):
                        # MSE loss
                        if weighing is not None:
                            reconstruct_loss = torch.mean((weighing.float(
                            ) * (b_0_reparam.float() - data_pts.float()) ** 2).reshape(data_pts.shape[0], -1), dim=1, )
                            reconstruct_loss = reconstruct_loss.mean()
                        else:
                            reconstruct_loss = F.mse_loss(
                                b_0_reparam.float(), data_pts.float(), reduction="mean")
                    else:
                        reconstruct_loss = torch.tensor(0.0)

                    if ("ns" in cfg.loss_type):
                        # compute diffusion loss
                        diffusion_loss = mse_loss(e, eps_theta)
                    else:
                        diffusion_loss = torch.tensor(0.0)

                    # total loss
                    loss = diffusion_loss + reconstruct_loss
                    loss = torch.nan_to_num(loss)

                    pbar.set_postfix(
                        {'diffusion': diffusion_loss.item(), 'reconstruct': reconstruct_loss.item()})

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        params_to_clip = model_ddpm_md.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, 1.0)

                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_train_loss += accelerator.gather(loss).mean().item()
                    if accelerator.is_main_process and (i % log_interval == 0):
                        log_and_write(epoch, loss, i, len(
                            train_loader), "train", summary_writer)

        mean_epoch_train_loss = epoch_train_loss / len(train_loader)
        if accelerator.is_main_process:
            log_and_write(epoch, mean_epoch_train_loss,
                          loss_type="train_epoch_mean", summary_writer=summary_writer)

    if accelerator.is_main_process:
        summary_writer.close()
