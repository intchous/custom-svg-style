import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from deepsvg.json_help import load_json
from svg_ldm.my_svg_dataset_svgs_ldm import SVGs_Dataset_noscale, split_train_test
from svg_ldm.svg_dataset_ldm import SVGs_Dataset_pts, SVGs_Dataset_pts_cont


def get_data_list(obj_sign="IconShopsvg_diffvg_select", signature="IconShopsvg_diffvg_select_inversionz", par_dir="./dataset/IconShop/", use_wh=True, max_paths_len_thresh=25, select_lbids_dict=None):

    svg_data_dir = os.path.join(par_dir, signature)
    svg_data_img_dir = svg_data_dir

    if (use_wh):
        json_fp = os.path.join(par_dir, obj_sign + "_wh.json")
    else:
        json_fp = os.path.join(par_dir, obj_sign + ".json")

    path_info = {}
    if (os.path.exists(json_fp)):
        path_info = load_json(json_fp)

    caption_fp = os.path.join(par_dir, "label.csv")
    caption_df = pd.read_csv(caption_fp)
    # 生成 id-label 的映射字典
    id_label_map = dict(zip(caption_df['id'].astype(
        int), caption_df['label'].astype(str)))

    tensor_file_list = []
    for _fn_pre in path_info.keys():
        if (("num_paths" in path_info[_fn_pre]) and ("seq_np_sv_fp" in path_info[_fn_pre])):
            num_paths = path_info[_fn_pre]["num_paths"]

            if ((num_paths > 0) and (num_paths < max_paths_len_thresh)):
                _fn = os.path.basename(path_info[_fn_pre]["seq_np_sv_fp"])
                img_fn = _fn.replace(".npz", ".png").replace("optm_", "")
                img_fp = os.path.join(svg_data_img_dir, img_fn)

                cur_id = img_fn.split("_")[0]
                cur_id_int = int(cur_id)
                label = id_label_map.get(cur_id_int, "")

                if (os.path.exists(img_fp) and (label != "")):
                    if (select_lbids_dict is None):
                        tensor_file_list.append(_fn)
                    else:
                        if (cur_id_int in select_lbids_dict):
                            tensor_file_list.append(_fn)

    split_res = split_train_test(tensor_file_list, train_ratio=0.99999)
    tensor_file_list_train = split_res["file_list_train"]
    tensor_file_list_test = split_res["file_list_test"]
    tensor_file_list_train.extend(tensor_file_list_test)

    img_fp_list_train = []
    caption_list_train = []
    for _fn in tensor_file_list_train:
        img_fn = _fn.replace(".npz", ".png").replace("optm_", "")
        img_fp = os.path.join(svg_data_img_dir, img_fn)
        img_fp_list_train.append(img_fp)

        cur_id = img_fn.split("_")[0]
        label = id_label_map.get(int(cur_id), "")
        label = label.replace("/", ", ")

        caption_list_train.append(label)

    img_fp_list_test = []
    caption_list_test = []
    for _fn in tensor_file_list_test:
        img_fn = _fn.replace(".npz", ".png").replace("optm_", "")
        img_fp = os.path.join(svg_data_img_dir, img_fn)
        img_fp_list_test.append(img_fp)

        cur_id = img_fn.split("_")[0]
        label = id_label_map.get(int(cur_id), "")
        label = label.replace("/", ", ")
        # print("label: ", label)
        caption_list_test.append(label)

    return svg_data_dir, tensor_file_list_train, tensor_file_list_test, caption_list_train, caption_list_test, img_fp_list_train, img_fp_list_test


def get_data_pts_list(obj_sign="IconShop_diffvgimg_merge_resize", signature="IconShop_diffvg_select", par_dir="./dataset/IconShop/", max_paths_len_thresh=25, max_points=1000, select_lbids_dict=None, img_cid_list=None, add_style_token=False):

    svg_data_dir = os.path.join(par_dir, signature)
    json_fp = os.path.join(par_dir, obj_sign + ".json")

    path_info = {}
    if (os.path.exists(json_fp)):
        path_info = load_json(json_fp)

    caption_fp = os.path.join(par_dir, "label.csv")
    caption_df = pd.read_csv(caption_fp)

    # 生成 id-label 的映射字典
    if ("IconShop" in signature):
        id_label_map = dict(zip(caption_df['id'].astype(
            int), caption_df['label'].astype(str)))
    else:
        id_label_map = dict(zip(caption_df['id'].astype(
            str), caption_df['label'].astype(str)))
        id_cid_map = dict(zip(caption_df['id'].astype(
            str), caption_df['cid'].astype(str)))

    tensor_file_list = []
    for _fn_pre in path_info.keys():

        if (("path_num" in path_info[_fn_pre]) and ("total_points" in path_info[_fn_pre])):
            num_paths = path_info[_fn_pre]["path_num"]
            total_points = path_info[_fn_pre]["total_points"]

            if ((num_paths > 0) and (num_paths < max_paths_len_thresh) and (total_points > 0) and (total_points < max_points)):

                _fn = os.path.basename(path_info[_fn_pre]["svg_fp"])
                if ("customsvg" in signature):
                    pickle_fn = _fn.replace(".svg", "_feat.pkl")
                else:
                    pickle_fn = _fn.replace(".svg", "_diffvg_rsz_feat.pkl")
                pickle_fp = os.path.join(svg_data_dir, pickle_fn)

                cur_id = os.path.splitext(_fn)[0]
                if ("IconShop" in signature):
                    cur_id_int = int(cur_id)
                else:
                    cur_id_int = str(cur_id)

                label = id_label_map.get(cur_id_int, "")

                if (os.path.exists(pickle_fp) and (label != "")):
                    flg_select_lbids = False
                    if (select_lbids_dict is None):
                        flg_select_lbids = True
                    else:
                        if (cur_id_int in select_lbids_dict):
                            flg_select_lbids = True

                    flg_img_cid = False
                    if (img_cid_list is None):
                        flg_img_cid = True
                    else:
                        cur_cid = id_cid_map.get(str(cur_id), "")
                        if (cur_cid in img_cid_list):
                            flg_img_cid = True

                    if (flg_select_lbids and flg_img_cid):
                        tensor_file_list.append(pickle_fn)

    split_res = split_train_test(tensor_file_list, train_ratio=0.99999)
    tensor_file_list_train = split_res["file_list_train"]
    tensor_file_list_test = split_res["file_list_test"]
    tensor_file_list_train.extend(tensor_file_list_test)

    caption_list_train = []
    for pickle_fn in tensor_file_list_train:
        if ("customsvg" in signature):
            _fn = pickle_fn.replace("_feat.pkl", ".svg")
        else:
            _fn = pickle_fn.replace("_diffvg_rsz_feat.pkl", ".svg")

        cur_id = os.path.splitext(_fn)[0]
        if ("IconShop" in signature):
            label = id_label_map.get(int(cur_id), "")
        else:
            label = id_label_map.get(str(cur_id), "")

        label = label.replace("/", ", ")
        # assert (label != "")

        if (add_style_token):
            if ("IconShop" in signature):
                style_sign = "; iconshop style"
            elif ("svgrepo" in signature):
                cur_cid = id_cid_map.get(str(cur_id), "")
                style_sign = f"; svgrepo_{cur_cid} style"
            elif ("customsvg" in signature):
                cur_cid = id_cid_map.get(str(cur_id), "")
                style_sign = f"; {cur_cid} style"
            else:
                cur_cid = id_cid_map.get(str(cur_id), "")
                style_sign = f"; iconfontcl_{cur_cid} style"

            label = label + style_sign

        caption_list_train.append(label)

    caption_list_test = []
    for pickle_fn in tensor_file_list_test:
        if ("customsvg" in signature):
            _fn = pickle_fn.replace("_feat.pkl", ".svg")
        else:
            _fn = pickle_fn.replace("_diffvg_rsz_feat.pkl", ".svg")

        cur_id = os.path.splitext(_fn)[0]
        if ("IconShop" in signature):
            label = id_label_map.get(int(cur_id), "")
        else:
            label = id_label_map.get(str(cur_id), "")
        label = label.replace("/", ", ")
        # assert (label != "")
        caption_list_train.append(label)

    return svg_data_dir, tensor_file_list_train, tensor_file_list_test, caption_list_train, caption_list_test


# ------------------------------------
def get_data_pts_dream_list(obj_sign="svgrepo_collections_diffvgimg_merge_resize", signature="svgrepo_collections_diffvg_select", par_dir="./dataset/svgrepo_collections/", max_paths_len_thresh=25, max_points=1000, cid_list=None):

    svg_data_dir = os.path.join(par_dir, signature)
    json_fp = os.path.join(par_dir, obj_sign + ".json")

    path_info = {}
    if (os.path.exists(json_fp)):
        path_info = load_json(json_fp)

    caption_fp = os.path.join(par_dir, "label.csv")
    caption_df = pd.read_csv(caption_fp)

    # 生成 id-label 的映射字典
    id_label_map = dict(zip(caption_df['id'].astype(
        str), caption_df['label'].astype(str)))
    id_cid_map = dict(zip(caption_df['id'].astype(
        str), caption_df['cid'].astype(str)))

    tensor_file_list = []
    for _fn_pre in path_info.keys():

        if (("path_num" in path_info[_fn_pre]) and ("total_points" in path_info[_fn_pre])):
            num_paths = path_info[_fn_pre]["path_num"]
            total_points = path_info[_fn_pre]["total_points"]

            if ((num_paths > 0) and (num_paths < max_paths_len_thresh) and (total_points > 0) and (total_points < max_points)):

                _fn = os.path.basename(path_info[_fn_pre]["svg_fp"])
                pickle_fn = _fn.replace(".svg", "_diffvg_rsz_feat.pkl")
                pickle_fp = os.path.join(svg_data_dir, pickle_fn)

                cur_id = os.path.splitext(_fn)[0]
                cur_id_int = str(cur_id)

                label = id_label_map.get(cur_id_int, "")
                if (os.path.exists(pickle_fp) and (label != "")):

                    if (cid_list is None):
                        tensor_file_list.append(pickle_fn)
                    else:
                        cur_cid = id_cid_map.get(cur_id_int, "")
                        if (cur_cid in cid_list):
                            tensor_file_list.append(pickle_fn)

    tensor_file_list_train = tensor_file_list
    tensor_file_list_test = [tensor_file_list[0]]

    caption_list_train = []
    for pickle_fn in tensor_file_list_train:
        _fn = pickle_fn.replace("_diffvg_rsz_feat.pkl", ".svg")

        cur_id = os.path.splitext(_fn)[0]
        if ("IconShop" in signature):
            label = id_label_map.get(int(cur_id), "")
        else:
            label = id_label_map.get(str(cur_id), "")

        label = label.replace("/", ", ")
        # assert (label != "")
        caption_list_train.append(label)

    caption_list_test = [caption_list_train[0]]

    return svg_data_dir, tensor_file_list_train, tensor_file_list_test, caption_list_train, caption_list_test


# ------------------------------------
def get_data_pts_imgs_list(obj_sign="IconShop_diffvgimg_merge_resize", signature="IconShop_diffvg_select", color_imgs_signature="IconShop_diffvg_select_color", par_dir="./dataset/IconShop/", max_paths_len_thresh=25, max_points=1000, select_lbids_dict=None):

    svg_data_dir = os.path.join(par_dir, signature)
    color_imgs_data_dir = os.path.join(par_dir, color_imgs_signature)
    json_fp = os.path.join(par_dir, obj_sign + ".json")

    path_info = {}
    if (os.path.exists(json_fp)):
        path_info = load_json(json_fp)

    caption_fp = os.path.join(par_dir, "label.csv")
    caption_df = pd.read_csv(caption_fp)

    # 生成 id-label 的映射字典
    if ("IconShop" in signature):
        id_label_map = dict(
            zip(caption_df['id'].astype(int), caption_df['label'].astype(str)))
    else:
        id_label_map = dict(zip(caption_df['id'].astype(
            str), caption_df['label'].astype(str)))

    tensor_file_list = []
    for _fn_pre in path_info.keys():

        if (("path_num" in path_info[_fn_pre]) and ("total_points" in path_info[_fn_pre])):
            num_paths = path_info[_fn_pre]["path_num"]
            total_points = path_info[_fn_pre]["total_points"]

            if ((num_paths > 0) and (num_paths < max_paths_len_thresh) and (total_points > 0) and (total_points < max_points)):

                _fn = os.path.basename(path_info[_fn_pre]["svg_fp"])
                pickle_fn = _fn.replace(".svg", "_diffvg_rsz_feat.pkl")
                pickle_fp = os.path.join(svg_data_dir, pickle_fn)

                img_fn = _fn.replace(".svg", "_diffvg_rsz_color.png")
                img_fp = os.path.join(color_imgs_data_dir, img_fn)

                cur_id = os.path.splitext(_fn)[0]
                if ("IconShop" in signature):
                    cur_id_int = int(cur_id)
                else:
                    cur_id_int = str(cur_id)

                label = id_label_map.get(cur_id_int, "")

                if (os.path.exists(pickle_fp) and os.path.exists(img_fp) and (label != "")):
                    if (select_lbids_dict is None):
                        tensor_file_list.append(pickle_fn)
                    else:
                        if (cur_id_int in select_lbids_dict):
                            tensor_file_list.append(pickle_fn)

    split_res = split_train_test(tensor_file_list, train_ratio=0.99999)
    tensor_file_list_train = split_res["file_list_train"]
    tensor_file_list_test = split_res["file_list_test"]
    tensor_file_list_train.extend(tensor_file_list_test)

    img_fp_list_train = []
    caption_list_train = []
    for pickle_fn in tensor_file_list_train:
        _fn = pickle_fn.replace("_diffvg_rsz_feat.pkl", ".svg")

        img_fn = _fn.replace(".svg", "_diffvg_rsz_color.png")
        img_fp = os.path.join(color_imgs_data_dir, img_fn)
        img_fp_list_train.append(img_fp)

        cur_id = os.path.splitext(_fn)[0]
        if ("IconShop" in signature):
            label = id_label_map.get(int(cur_id), "")
        else:
            label = id_label_map.get(str(cur_id), "")
        label = label.replace("/", ", ")
        # assert (label != "")
        caption_list_train.append(label)

    img_fp_list_test = []
    caption_list_test = []
    for pickle_fn in tensor_file_list_test:
        _fn = pickle_fn.replace("_diffvg_rsz_feat.pkl", ".svg")

        img_fn = _fn.replace(".svg", "_diffvg_rsz_color.png")
        img_fp = os.path.join(color_imgs_data_dir, img_fn)
        img_fp_list_test.append(img_fp)

        cur_id = os.path.splitext(_fn)[0]
        if ("IconShop" in signature):
            label = id_label_map.get(int(cur_id), "")
        else:
            label = id_label_map.get(str(cur_id), "")
        label = label.replace("/", ", ")
        # assert (label != "")
        caption_list_train.append(label)

    return svg_data_dir, tensor_file_list_train, tensor_file_list_test, caption_list_train, caption_list_test, img_fp_list_train, img_fp_list_test


# ------------------------------------
def get_data_imgs_list(signature="iconfont_collections_diffvg_select", imgs_data_dir="./dataset", par_dir="./dataset", check_empty=False, min_num_colors=0):

    caption_fp = os.path.join(par_dir, "label.csv")
    caption_df = pd.read_csv(caption_fp)

    # 生成 id-label 的映射字典
    if ("IconShop" in signature):
        id_label_map = dict(
            zip(caption_df['id'].astype(int), caption_df['label'].astype(str)))
    else:
        id_label_map = dict(zip(caption_df['id'].astype(
            str), caption_df['label'].astype(str)))

        id_colors_map = dict(zip(caption_df['id'].astype(
            str), caption_df['num_colors'].astype(int)))

    imgs_list_train = []
    caption_list_train = []
    imgs_data_list = os.listdir(imgs_data_dir)
    for img_fn in imgs_data_list:
        if (not img_fn.endswith(".png")):
            continue

        _fn = img_fn.replace("_diffvg_rsz.png", ".svg")

        cur_id = os.path.splitext(_fn)[0]
        if ("IconShop" in signature):
            label = id_label_map.get(int(cur_id), "")
        else:
            label = id_label_map.get(str(cur_id), "")

        label = label.replace("/", ", ")
        # assert (label != "")

        num_colors = id_colors_map.get(str(cur_id), 0)
        if (num_colors >= min_num_colors):
            if (not check_empty or label):
                imgs_list_train.append(img_fn)
                caption_list_train.append(label)

    return imgs_list_train, caption_list_train


# ------------------------------------
def get_select_lbids_dict(cfg, par_dir="./dataset/IconShop/"):
    similarity_threshold = str(cfg.similarity_threshold)
    threshold_info_fp = os.path.join(
        par_dir, f"threshold_info_sim{similarity_threshold}.json")
    threshold_info = load_json(threshold_info_fp)

    # ----------------------------
    # select_threshold = 1050
    # select_threshold = 1000
    # Threshold >= 1000: Total Samples = 66415, Number of Clusters = 50
    # Threshold >= 1050: Total Samples = 59256, Number of Clusters = 43
    # select_threshold = 500
    select_threshold = str(cfg.select_threshold)

    select_lbids_dict = {}
    for itm in threshold_info[select_threshold]:
        # 注意这里itm['id']应该统一为int
        select_lbids_dict[int(itm['id'])] = 1

    return select_lbids_dict


# ------------------------------------
def get_dataset_pts(cfg, tensor_file_list_train=[], tensor_file_list_test=[], caption_list_train=[], caption_list_test=[], signature="IconShop_diffvg_select", svg_data_dir="./"):

    # svg_data_dir = os.path.join(par_dir, signature)
    # if (("_color" in signature) or ("iconfont_collections" in signature) or ("_conti" in signature)) SVGs_Dataset_pts_cont

    if (("IconShop" in signature) and ("_color" not in signature) and ("_conti" not in signature)):
        svg_dataset_train = SVGs_Dataset_pts(
            directory=svg_data_dir, max_paths=cfg.max_paths_len_thresh, max_points=cfg.max_points, file_list=tensor_file_list_train, caption_list=caption_list_train, use_cont_path_idx=cfg.use_cont_path_idx, only_black_white_color=cfg.only_black_white_color)
        svg_dataset_test = SVGs_Dataset_pts(
            directory=svg_data_dir, max_paths=cfg.max_paths_len_thresh, max_points=cfg.max_points, file_list=tensor_file_list_test, caption_list=caption_list_test, use_cont_path_idx=cfg.use_cont_path_idx, only_black_white_color=cfg.only_black_white_color)
    else:
        svg_dataset_train = SVGs_Dataset_pts_cont(
            directory=svg_data_dir, max_paths=cfg.max_paths_len_thresh, max_points=cfg.max_points, file_list=tensor_file_list_train, caption_list=caption_list_train, only_black_white_color=cfg.only_black_white_color)
        svg_dataset_test = SVGs_Dataset_pts_cont(
            directory=svg_data_dir, max_paths=cfg.max_paths_len_thresh, max_points=cfg.max_points, file_list=tensor_file_list_test, caption_list=caption_list_test, only_black_white_color=cfg.only_black_white_color)

    return svg_dataset_train, svg_dataset_test


def get_dataset_latents(cfg, tensor_file_list_train=[], tensor_file_list_test=[], caption_list_train=[], caption_list_test=[], svg_data_dir="./", h=224, w=224):

    # svg_data_dir = os.path.join(par_dir, signature)

    svg_dataset_train = SVGs_Dataset_noscale(
        directory=svg_data_dir, h=h, w=w, fixed_paths=cfg.max_total_len, file_list=tensor_file_list_train, caption_list=caption_list_train)
    svg_dataset_test = SVGs_Dataset_noscale(
        directory=svg_data_dir, h=h, w=w, fixed_paths=cfg.max_total_len, file_list=tensor_file_list_test, caption_list=caption_list_test)

    return svg_dataset_train, svg_dataset_test


def get_dataloaders(cfg, data_type="pts", tensor_file_list_train=[], tensor_file_list_test=[], caption_list_train=[], caption_list_test=[], signature="IconShop_diffvg_select", svg_data_dir="./", h=224, w=224):

    # svg_data_dir = os.path.join(par_dir, signature)

    if (data_type == "pts"):
        svg_dataset_train, svg_dataset_test = get_dataset_pts(cfg=cfg, tensor_file_list_train=tensor_file_list_train, tensor_file_list_test=tensor_file_list_test,
                                                              caption_list_train=caption_list_train, caption_list_test=caption_list_test, signature=signature, svg_data_dir=svg_data_dir)
    else:
        svg_dataset_train, svg_dataset_test = get_dataset_latents(cfg=cfg, tensor_file_list_train=tensor_file_list_train, tensor_file_list_test=tensor_file_list_test,
                                                                  caption_list_train=caption_list_train, caption_list_test=caption_list_test, svg_data_dir=svg_data_dir, h=h, w=w)
    # ----------------------------
    # persistent_workers=True
    train_loader = DataLoader(svg_dataset_train, batch_size=cfg.batch_size,
                              num_workers=cfg.loader_num_workers, shuffle=True)
    test_loader = DataLoader(svg_dataset_test, batch_size=cfg.batch_size,
                             num_workers=cfg.loader_num_workers, shuffle=False)

    return train_loader, test_loader
