import os
import random
import numpy as np
import pandas as pd
import pickle
import PIL

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def binary_tensor_to_decimal(tensors):
    binary_tensors = (tensors >= 0.5).int()
    num_bits = binary_tensors.shape[-1]
    weights = 2 ** torch.arange(num_bits - 1, -1, -1,
                                dtype=torch.int, device=tensors.device)

    decimal_values = torch.sum(binary_tensors * weights, dim=-1)

    return decimal_values


def load_target(fp, size=(224, 224), img_channel=3, return_rgb=False):
    target = PIL.Image.open(fp)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = PIL.Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")

    if target.size != size:
        target = target.resize(size, PIL.Image.Resampling.BICUBIC)

    if (return_rgb):
        return target

    transforms_ = [transforms.ToTensor(), transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]

    data_transforms = transforms.Compose(transforms_)  # w,h,c -> c,h,w
    gt = data_transforms(target)

    if img_channel == 1:
        gt = gt.mean(dim=0, keepdim=True)

    return gt


# ----------------------------------------
class SVGs_Dataset_pts(Dataset):
    def __init__(self, directory, max_paths=30, max_points=1000, file_list=None, caption_list=None, use_cont_path_idx=False, only_black_white_color=False):
        super(SVGs_Dataset_pts, self).__init__()
        self.directory = directory

        if file_list is None:
            self.file_list = os.listdir(self.directory)
        else:
            self.file_list = file_list

        self.max_paths = max_paths
        self.num_bits = (max(30, self.max_paths)).bit_length()
        self.max_path_idx = self.max_paths

        self.max_points = max_points
        self.caption_list = caption_list
        self.use_cont_path_idx = use_cont_path_idx
        self.only_black_white_color = only_black_white_color

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.directory, self.file_list[idx])

        # ---------------------------------------------
        with open(filepath, 'rb') as f:
            tp_data = pickle.load(f)

        pts_feat_tensor = tp_data['pts_feat_tensor']

        if (self.only_black_white_color):
            rgba_tensor = pts_feat_tensor[:, -4:]  # [n, 4]
            rgb_avg = rgba_tensor[:, :3].mean(dim=1)
            binary_rgb = torch.where(rgb_avg > 0.5, 1.0, 0.0)
            pts_feat_tensor = torch.cat(
                (pts_feat_tensor[:, :-4], binary_rgb.unsqueeze(1)), dim=1)

        # ---------------------------------------
        act_num_bits = self.num_bits
        if (self.use_cont_path_idx):
            bits_tensors = pts_feat_tensor[:, :self.num_bits]
            decimal_values = binary_tensor_to_decimal(bits_tensors)
            normalized_decimal_values = decimal_values * 1.0 / self.max_path_idx

            pts_feat_tensor = torch.cat((normalized_decimal_values.unsqueeze(
                1), pts_feat_tensor[:, self.num_bits:]), dim=1)

            act_num_bits = 1
        # ---------------------------------------

        total_points = pts_feat_tensor.shape[0]
        assert total_points < self.max_points

        # ---------------------------------------------
        padding_needed = max(0, self.max_points - total_points)
        padding_tensor = torch.zeros(
            (padding_needed, pts_feat_tensor.shape[1]))
        padding_tensor[:, :act_num_bits] = 1.0

        padded_pts_tensor = torch.cat([pts_feat_tensor, padding_tensor], dim=0)
        p_mask = torch.zeros((self.max_points, pts_feat_tensor.shape[1]))
        p_mask[:total_points, :] = 1.0

        # -------------------------------
        caption = ""
        if self.caption_list:
            caption = self.caption_list[idx]

        res_data = {
            "padded_pts_tensor": padded_pts_tensor,
            "total_points_ad1": total_points + 1,
            "filepaths": filepath,
            "p_mask": p_mask,
            "caption": caption
        }
        return res_data
# ----------------------------------------


# ----------------------------------------
class SVGs_Dataset_pts_imgs(Dataset):
    def __init__(self, directory, max_paths=30, max_points=1000, file_list=None, caption_list=None, img_list=None, use_cont_path_idx=False, only_black_white_color=False):
        super(SVGs_Dataset_pts_imgs, self).__init__()
        self.directory = directory

        if file_list is None:
            self.file_list = os.listdir(self.directory)
        else:
            self.file_list = file_list

        self.max_paths = max_paths
        self.num_bits = (max(30, self.max_paths)).bit_length()
        self.max_path_idx = self.max_paths

        self.max_points = max_points
        self.caption_list = caption_list
        self.img_list = img_list
        self.use_cont_path_idx = use_cont_path_idx
        self.only_black_white_color = only_black_white_color

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.directory, self.file_list[idx])

        # ---------------------------------------------
        with open(filepath, 'rb') as f:
            tp_data = pickle.load(f)

        pts_feat_tensor = tp_data['pts_feat_tensor']

        if (self.only_black_white_color):
            rgba_tensor = pts_feat_tensor[:, -4:]  # [n, 4]
            rgb_avg = rgba_tensor[:, :3].mean(dim=1)
            binary_rgb = torch.where(rgb_avg > 0.5, 1.0, 0.0)
            pts_feat_tensor = torch.cat(
                (pts_feat_tensor[:, :-4], binary_rgb.unsqueeze(1)), dim=1)

        # ---------------------------------------
        act_num_bits = self.num_bits
        if (self.use_cont_path_idx):
            bits_tensors = pts_feat_tensor[:, :self.num_bits]
            decimal_values = binary_tensor_to_decimal(bits_tensors)
            normalized_decimal_values = decimal_values * 1.0 / self.max_path_idx

            pts_feat_tensor = torch.cat((normalized_decimal_values.unsqueeze(
                1), pts_feat_tensor[:, self.num_bits:]), dim=1)

            act_num_bits = 1
        # ---------------------------------------

        total_points = pts_feat_tensor.shape[0]
        assert total_points < self.max_points

        # ---------------------------------------------
        padding_needed = max(0, self.max_points - total_points)
        padding_tensor = torch.zeros(
            (padding_needed, pts_feat_tensor.shape[1]))
        padding_tensor[:, :act_num_bits] = 1.0

        padded_pts_tensor = torch.cat([pts_feat_tensor, padding_tensor], dim=0)
        p_mask = torch.zeros((self.max_points, pts_feat_tensor.shape[1]))
        p_mask[:total_points, :] = 1.0

        # -------------------------------
        caption = ""
        if self.caption_list:
            caption = self.caption_list[idx]

        # -------------------------------
        path_img = []
        if self.img_list:
            im_path = self.img_list[idx]
            assert os.path.exists(im_path)

            if (os.path.exists(im_path)):
                path_img = load_target(fp=im_path)
        # -------------------------------

        res_data = {
            "padded_pts_tensor": padded_pts_tensor,
            "total_points_ad1": total_points + 1,
            "filepaths": filepath,
            "p_mask": p_mask,
            "caption": caption,
            "path_img": path_img
        }
        return res_data
# ----------------------------------------


# ----------------------------------------
class SVGs_Dataset_pts_cont(Dataset):
    def __init__(self, directory, max_paths=30, max_points=1000, file_list=None, caption_list=None, only_black_white_color=False):
        super(SVGs_Dataset_pts_cont, self).__init__()
        self.directory = directory

        if file_list is None:
            self.file_list = os.listdir(self.directory)
        else:
            self.file_list = file_list

        self.max_paths = max_paths
        self.num_bits = 1
        self.max_path_idx = self.max_paths

        self.max_points = max_points
        self.caption_list = caption_list
        self.only_black_white_color = only_black_white_color

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.directory, self.file_list[idx])

        # ---------------------------------------------
        with open(filepath, 'rb') as f:
            tp_data = pickle.load(f)

        pts_feat_tensor = tp_data['pts_feat_tensor']

        if (self.only_black_white_color):
            rgba_tensor = pts_feat_tensor[:, -4:]  # [n, 4]
            rgb_avg = rgba_tensor[:, :3].mean(dim=1)
            binary_rgb = torch.where(rgb_avg > 0.5, 1.0, 0.0)
            pts_feat_tensor = torch.cat(
                (pts_feat_tensor[:, :-4], binary_rgb.unsqueeze(1)), dim=1)

        # ---------------------------------------
        act_num_bits = self.num_bits

        decimal_values = pts_feat_tensor[:, :self.num_bits]

        normalized_decimal_values = decimal_values * 1.0 / self.max_path_idx

        pts_feat_tensor = torch.cat(
            (normalized_decimal_values, pts_feat_tensor[:, self.num_bits:]), dim=1)

        # ---------------------------------------

        total_points = pts_feat_tensor.shape[0]
        assert total_points < self.max_points

        # ---------------------------------------------
        padding_needed = max(0, self.max_points - total_points)
        padding_tensor = torch.zeros(
            (padding_needed, pts_feat_tensor.shape[1]))
        padding_tensor[:, :act_num_bits] = 1.0

        padded_pts_tensor = torch.cat([pts_feat_tensor, padding_tensor], dim=0)
        p_mask = torch.zeros((self.max_points, pts_feat_tensor.shape[1]))
        p_mask[:total_points, :] = 1.0

        # -------------------------------
        caption = ""
        if self.caption_list:
            caption = self.caption_list[idx]

        res_data = {
            "padded_pts_tensor": padded_pts_tensor,
            "total_points_ad1": total_points + 1,
            "filepaths": filepath,
            "p_mask": p_mask,
            "caption": caption
        }
        return res_data
# ----------------------------------------


# ----------------------------------------
class SVGs_Dataset_pts_imgs_cont(Dataset):
    def __init__(self, directory, max_paths=30, max_points=1000, file_list=None, caption_list=None, img_list=None, only_black_white_color=False):
        super(SVGs_Dataset_pts_imgs_cont, self).__init__()
        self.directory = directory

        if file_list is None:
            self.file_list = os.listdir(self.directory)
        else:
            self.file_list = file_list

        self.max_paths = max_paths
        self.num_bits = 1
        self.max_path_idx = self.max_paths

        self.max_points = max_points
        self.caption_list = caption_list
        self.img_list = img_list
        self.only_black_white_color = only_black_white_color

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filepath = os.path.join(self.directory, self.file_list[idx])

        # ---------------------------------------------
        with open(filepath, 'rb') as f:
            tp_data = pickle.load(f)

        pts_feat_tensor = tp_data['pts_feat_tensor']

        if (self.only_black_white_color):
            rgba_tensor = pts_feat_tensor[:, -4:]  # [n, 4]
            rgb_avg = rgba_tensor[:, :3].mean(dim=1)
            binary_rgb = torch.where(rgb_avg > 0.5, 1.0, 0.0)
            pts_feat_tensor = torch.cat(
                (pts_feat_tensor[:, :-4], binary_rgb.unsqueeze(1)), dim=1)

        # ---------------------------------------
        act_num_bits = self.num_bits

        decimal_values = pts_feat_tensor[:, :self.num_bits]

        normalized_decimal_values = decimal_values * 1.0 / self.max_path_idx

        pts_feat_tensor = torch.cat(
            (normalized_decimal_values, pts_feat_tensor[:, self.num_bits:]), dim=1)

        # ---------------------------------------

        total_points = pts_feat_tensor.shape[0]
        assert total_points < self.max_points

        # ---------------------------------------------
        padding_needed = max(0, self.max_points - total_points)
        padding_tensor = torch.zeros(
            (padding_needed, pts_feat_tensor.shape[1]))
        padding_tensor[:, :act_num_bits] = 1.0

        padded_pts_tensor = torch.cat([pts_feat_tensor, padding_tensor], dim=0)
        p_mask = torch.zeros((self.max_points, pts_feat_tensor.shape[1]))
        p_mask[:total_points, :] = 1.0

        # -------------------------------
        caption = ""
        if self.caption_list:
            caption = self.caption_list[idx]

        # -------------------------------
        path_img = []
        if self.img_list:
            im_path = self.img_list[idx]
            assert os.path.exists(im_path)

            if (os.path.exists(im_path)):
                path_img = load_target(fp=im_path)
        # -------------------------------

        res_data = {
            "padded_pts_tensor": padded_pts_tensor,
            "total_points_ad1": total_points + 1,
            "filepaths": filepath,
            "p_mask": p_mask,
            "caption": caption,
            "path_img": path_img
        }
        return res_data
# ----------------------------------------


# ----------------------------------------
class SVGs_Dataset_imgs(Dataset):
    def __init__(self, caption_list=None, img_list=None, img_size=(224, 224), img_channel=3):
        super(SVGs_Dataset_imgs, self).__init__()

        self.caption_list = caption_list
        self.img_list = img_list
        self.img_size = img_size
        self.img_channel = img_channel

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        caption = ""
        if self.caption_list:
            caption = self.caption_list[idx]

        # -------------------------------
        path_img = []
        im_path = ""
        if self.img_list:
            im_path = self.img_list[idx]
            assert os.path.exists(im_path)

            if (os.path.exists(im_path)):
                path_img = load_target(
                    fp=im_path, size=self.img_size, img_channel=self.img_channel)
        # -------------------------------

        res_data = {
            "im_path": im_path,
            "caption": caption,
            "path_img": path_img
        }
        return res_data
# ----------------------------------------


def split_train_test(file_list, train_ratio=0.9):
    # split train and test
    random.shuffle(file_list)
    len_train = int(len(file_list)*train_ratio)
    file_list_train = file_list[:len_train]
    file_list_test = file_list[len_train:]

    return {
        "file_list_train": file_list_train,
        "file_list_test": file_list_test
    }


def save_filelist_csv(file_list, file_list_fp):
    file_list_df = pd.DataFrame(file_list, columns=["file_name"])
    file_list_df.to_csv(file_list_fp, index=False)
