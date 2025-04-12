import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import nibabel as nib

class NodeDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_suffixes, transforms=None, 
                 cls_features=None, reg_features=None, inference_mode=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_suffixes = image_suffixes  # 直接传入该节点需要的图像后缀列表
        self.transforms = transforms or []    # 直接接收该节点的变换列表
        self.inference_mode = inference_mode
        self.cls_features = cls_features or []
        self.reg_features = reg_features or []

        # 获取所有图像文件
        image_files = sorted(os.listdir(image_dir))
        # 提取 case_ids，假设文件名格式为 case_{case_id}_{suffix}.nii.gz
        self.case_ids = sorted(set(f.split('_')[1] for f in image_files if f.endswith('.nii.gz')))
        self.image_files = image_files

        # 验证每个 case 是否有 image_suffixes 中指定的后缀图像
        required_suffixes = set(self.image_suffixes)
        for case_id in self.case_ids:
            available_images = [f for f in image_files if f.startswith(f'case_{case_id}_')]
            available_suffixes = set(f.split('_')[2].replace('.nii.gz', '') for f in available_images)
            missing_suffixes = required_suffixes - available_suffixes
            if missing_suffixes:
                raise FileNotFoundError(f"Case {case_id} missing images with suffixes {missing_suffixes}")

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        images = []

        # 加载该节点的所有图像
        for suffix in self.image_suffixes:
            image_path = os.path.join(self.image_dir, f'case_{case_id}_{suffix}.nii.gz')
            image = nib.load(image_path).get_fdata()
            images.append(image)

        # 如果有图像，堆叠并应用变换
        if images:
            image_array = np.stack(images, axis=0)
            for t in self.transforms:
                image_array = t(image_array)
            image_tensor = torch.tensor(image_array, dtype=torch.float32)
        else:
            raise ValueError(f"No images loaded for case {case_id}")

        # 加载标签
        if not self.inference_mode:
            label_path = os.path.join(self.label_dir, f'case_{case_id}.csv')
            df = pd.read_csv(label_path)

            for feature in self.cls_features + self.reg_features:
                if feature not in df.columns:
                    raise ValueError(f"Label file {label_path} does not have column {feature}")

            labels = {}
            for feature in self.cls_features:
                values = df[feature].dropna().values
                if len(values) != 1:
                    raise ValueError(f"Expected 1 value for {feature} in {label_path}, got {len(values)}")
                try:
                    label_value = int(float(values[0]))
                    labels[feature] = torch.tensor(label_value, dtype=torch.long)
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid classification label for {feature} in {label_path}: {values[0]}")

            for feature in self.reg_features:
                values = df[feature].dropna().values
                if len(values) != 1:
                    raise ValueError(f"Expected 1 value for {feature} in {label_path}, got {len(values)}")
                labels[feature] = torch.tensor(float(values[0]), dtype=torch.float32)
        else:
            labels = {feature: torch.tensor(0, dtype=torch.long if feature in self.cls_features else torch.float32)
                      for feature in self.cls_features + self.reg_features}

        return image_tensor, labels
