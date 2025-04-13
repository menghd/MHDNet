import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import LambdaLR
from copy import deepcopy

sys.path.append(r"C:\Users\PC\PycharmProjects\thu_xwh\Codes")
from node_toolkit.mhdnet import MHDNet, HDNet
from node_toolkit.node_dataset import NodeDataset
from node_toolkit.node_loss import focal_loss, mse_loss
from node_toolkit.node_utils import train, validate
from node_toolkit.node_transform import (
    MinMaxNormalize3D,
    ZScoreNormalize3D,
    RandomRotate3D,
    RandomFlip3D,
    RandomShift3D,
    RandomZoom3D,
)

def main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 数据路径
    image_dir = "C:/Users/PC/PycharmProjects/thu_xwh/Data/TrainNiigzCsvData/imagesTr"
    label_dir = "C:/Users/PC/PycharmProjects/thu_xwh/Data/TrainNiigzCsvData/labelsTr"

    # 保存路径
    save_dir = "C:/Users/PC/PycharmProjects/thu_xwh/Codes/Model/MHDNet0412"
    os.makedirs(save_dir, exist_ok=True)

    # 超参数
    num_dimensions = 3
    batch_size = 8  # 注意：加深网络可能增加显存需求，必要时可减小到 4
    num_epochs = 200
    learning_rate = 1e-3
    k_folds = 5
    validation_interval = 1
    patience = 200

    # 子网络1（粗处理，保持不变）
    node_configs1 = {
        0: (1, 64, 64, 64),  
        1: (1, 64, 64, 64), 
        2: (1, 64, 64, 64), 
        3: (1, 64, 64, 64), 
        4: (1, 64, 64, 64), 
        5: (32, 64, 64, 64),  
        6: (32, 64, 64, 64),
        7: (32, 64, 64, 64),
        8: (32, 64, 64, 64),
        9: (32, 64, 64, 64),
        10: (32, 64, 64, 64),
        11: (64, 32, 32, 32),
    }
    hyperedge_configs1 = {
        "e1": {
            "src_nodes": [0, 1, 2, 3, 4],
            "dst_nodes": [5],
            "params": {
                "convs": [(32, 5, 5, 5), (32, 5, 5, 5)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
            },
        },
        "e2": {
            "src_nodes": [0],
            "dst_nodes": [6],
            "params": {
                "convs": [(32, 5, 5, 5), (32, 5, 5, 5)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
            },
        },
        "e3": {
            "src_nodes": [1],
            "dst_nodes": [7],
            "params": {
                "convs": [(32, 5, 5, 5), (32, 5, 5, 5)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
            },
        },
        "e4": {
            "src_nodes": [2],
            "dst_nodes": [8],
            "params": {
                "convs": [(32, 5, 5, 5), (32, 5, 5, 5)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
            },
        },
        "e5": {
            "src_nodes": [3],
            "dst_nodes": [9],
            "params": {
                "convs": [(32, 5, 5, 5), (32, 5, 5, 5)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
            },
        },
        "e6": {
            "src_nodes": [4],
            "dst_nodes": [10],
            "params": {
                "convs": [(32, 5, 5, 5), (32, 5, 5, 5)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
            },
        },
        "e7": {
            "src_nodes": [6, 7, 8, 9, 10],
            "dst_nodes": [11],
            "params": {
                "convs": [(64, 3, 3, 3), (64, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
            },
        },
        "e8": {
            "src_nodes": [5],
            "dst_nodes": [11],
            "params": {
                "convs": [(64, 3, 3, 3), (64, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (64, 64, 64),
                "out_p": "max"
            },
        },
    }
    in_nodes1 = [0, 1, 2, 3, 4]
    out_nodes1 = [11]

    # 子网络2（细处理，模仿 ResNet-18 加深到 8 个残差块）
    # 修改说明：
    # - 原结构：4 节点，逐步下采样。
    # - 新结构：4 个阶段，每个阶段 2 个残差块，共 8 块（16 层卷积），接近 ResNet-18 深度。
    # - 阶段划分：
    #   - 阶段1：节点 0-4，64 通道，32x32x32，2 个残差块
    #   - 阶段2：节点 5-9，128 通道，16x16x16，2 个残差块
    #   - 阶段3：节点 10-14，256 通道，8x8x8，2 个残差块
    #   - 阶段4：节点 15-19，512 通道，4x4x4，2 个残差块
    # - 输出节点 19 保持 (512, 4, 4, 4)，与任务分支兼容。
    node_configs2 = {
        0: (64, 32, 32, 32),   # 输入，连接子网络1
        # 阶段1：64 通道，32x32x32
        1: (64, 32, 32, 32),   # 残差块1
        2: (64, 32, 32, 32),   # 残差块1 输出
        3: (64, 32, 32, 32),   # 残差块2
        4: (64, 32, 32, 32),   # 残差块2 输出
        # 阶段2：128 通道，16x16x16
        5: (128, 16, 16, 16),  # 残差块3（下采样）
        6: (128, 16, 16, 16),  # 残差块3 输出
        7: (128, 16, 16, 16),  # 残差块4
        8: (128, 16, 16, 16),  # 残差块4 输出
        # 阶段3：256 通道，8x8x8
        9: (256, 8, 8, 8),     # 残差块5（下采样）
        10: (256, 8, 8, 8),    # 残差块5 输出
        11: (256, 8, 8, 8),    # 残差块6
        12: (256, 8, 8, 8),    # 残差块6 输出
        # 阶段4：512 通道，4x4x4
        13: (512, 4, 4, 4),    # 残差块7（下采样）
        14: (512, 4, 4, 4),    # 残差块7 输出
        15: (512, 4, 4, 4),    # 残差块8
        16: (512, 4, 4, 4),    # 残差块8 输出
        # 输出调整
        17: (512, 4, 4, 4),    # 最终输出
    }
    hyperedge_configs2 = {
        # 阶段1，残差块1：0 -> 1 -> 2，0 到 2 跳跃
        "e1": {
            "src_nodes": [0],
            "dst_nodes": [1],
            "params": {
                "convs": [(64, 3, 3, 3), (64, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (32, 32, 32),
            },
        },
        "e2": {
            "src_nodes": [1],
            "dst_nodes": [2],
            "params": {
                "convs": [(64, 3, 3, 3), (64, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (32, 32, 32),
            },
        },
        "e3": {
            "src_nodes": [0],
            "dst_nodes": [2],
            "params": {
                "convs": [],  # 跳跃连接，直接相加
                "feature_size": (32, 32, 32),
            },
        },
        # 残差块2：2 -> 3 -> 4，2 到 4 跳跃
        "e4": {
            "src_nodes": [2],
            "dst_nodes": [3],
            "params": {
                "convs": [(64, 3, 3, 3), (64, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (32, 32, 32),
            },
        },
        "e5": {
            "src_nodes": [3],
            "dst_nodes": [4],
            "params": {
                "convs": [(64, 3, 3, 3), (64, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (32, 32, 32),
            },
        },
        "e6": {
            "src_nodes": [2],
            "dst_nodes": [4],
            "params": {
                "convs": [],  # 跳跃连接，直接相加
                "feature_size": (32, 32, 32),
            },
        },
        # 阶段2，残差块3：4 -> 5 -> 6，4 到 6 跳跃（下采样）
        "e7": {
            "src_nodes": [4],
            "dst_nodes": [5],
            "params": {
                "convs": [(128, 3, 3, 3), (128, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (32, 32, 32),
                "out_p": 2,
            },
        },
        "e8": {
            "src_nodes": [5],
            "dst_nodes": [6],
            "params": {
                "convs": [(128, 3, 3, 3), (128, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (16, 16, 16),
            },
        },
        "e9": {
            "src_nodes": [4],
            "dst_nodes": [6],
            "params": {
                "convs": [(128, 1, 1, 1)],  # 跳跃连接，调整通道
                "norms": ["batch"],
                "feature_size": (32, 32, 32),
                "out_p": 2,
            },
        },
        # 残差块4：6 -> 7 -> 8，6 到 8 跳跃
        "e10": {
            "src_nodes": [6],
            "dst_nodes": [7],
            "params": {
                "convs": [(128, 3, 3, 3), (128, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (16, 16, 16),
            },
        },
        "e11": {
            "src_nodes": [7],
            "dst_nodes": [8],
            "params": {
                "convs": [(128, 3, 3, 3), (128, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (16, 16, 16),
            },
        },
        "e12": {
            "src_nodes": [6],
            "dst_nodes": [8],
            "params": {
                "convs": [],  # 跳跃连接，直接相加
                "feature_size": (16, 16, 16),
            },
        },
        # 阶段3，残差块5：8 -> 9 -> 10，8 到 10 跳跃（下采样）
        "e13": {
            "src_nodes": [8],
            "dst_nodes": [9],
            "params": {
                "convs": [(256, 3, 3, 3), (256, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (16, 16, 16),
                "out_p": 2,
            },
        },
        "e14": {
            "src_nodes": [9],
            "dst_nodes": [10],
            "params": {
                "convs": [(256, 3, 3, 3), (256, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (8, 8, 8),
            },
        },
        "e15": {
            "src_nodes": [8],
            "dst_nodes": [10],
            "params": {
                "convs": [(256, 1, 1, 1)],  # 跳跃连接，调整通道
                "norms": ["batch"],
                "feature_size": (16, 16, 16),
                "out_p": 2,
            },
        },
        # 残差块6：10 -> 11 -> 12，10 到 12 跳跃
        "e16": {
            "src_nodes": [10],
            "dst_nodes": [11],
            "params": {
                "convs": [(256, 3, 3, 3), (256, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (8, 8, 8),
            },
        },
        "e17": {
            "src_nodes": [11],
            "dst_nodes": [12],
            "params": {
                "convs": [(256, 3, 3, 3), (256, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (8, 8, 8),
            },
        },
        "e18": {
            "src_nodes": [10],
            "dst_nodes": [12],
            "params": {
                "convs": [],  # 跳跃连接，直接相加
                "feature_size": (8, 8, 8),
            },
        },
        # 阶段4，残差块7：12 -> 13 -> 14，12 到 14 跳跃（下采样）
        "e19": {
            "src_nodes": [12],
            "dst_nodes": [13],
            "params": {
                "convs": [(512, 3, 3, 3), (512, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (8, 8, 8),
                "out_p": 2,
            },
        },
        "e20": {
            "src_nodes": [13],
            "dst_nodes": [14],
            "params": {
                "convs": [(512, 3, 3, 3), (512, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (4, 4, 4),
            },
        },
        "e21": {
            "src_nodes": [12],
            "dst_nodes": [14],
            "params": {
                "convs": [(512, 1, 1, 1)],  # 跳跃连接，调整通道
                "norms": ["batch"],
                "feature_size": (8, 8, 8),
                "out_p": 2,
            },
        },
        # 残差块8：14 -> 15 -> 16，14 到 16 跳跃
        "e22": {
            "src_nodes": [14],
            "dst_nodes": [15],
            "params": {
                "convs": [(512, 3, 3, 3), (512, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (4, 4, 4),
            },
        },
        "e23": {
            "src_nodes": [15],
            "dst_nodes": [16],
            "params": {
                "convs": [(512, 3, 3, 3), (512, 3, 3, 3)],
                "norms": ["batch", "batch"],
                "acts": ["relu", "relu"],
                "feature_size": (4, 4, 4),
            },
        },
        "e24": {
            "src_nodes": [14],
            "dst_nodes": [16],
            "params": {
                "convs": [],  # 跳跃连接，直接相加
                "feature_size": (4, 4, 4),
            },
        },
        # 最终输出：16 -> 17
        "e25": {
            "src_nodes": [16],
            "dst_nodes": [17],
            "params": {
                "convs": [(512, 3, 3, 3)],
                "norms": ["batch"],
                "acts": ["relu"],
                "feature_size": (4, 4, 4),
            },
        },
    }
    in_nodes2 = [0]
    out_nodes2 = [17]  # 输出节点为 17，形状 (512, 4, 4, 4)

    # 子网络3（回归任务，保持不变）
    node_configs3 = {
        0: (512, 4, 4, 4),  # 输入，连接子网络2
        1: (1, 1, 1, 1),   # 输出：回归任务
    }
    hyperedge_configs3 = {
        "e1": {
            "src_nodes": [0],
            "dst_nodes": [1],
            "params": {
                "feature_size": (1, 1, 1),
                "convs": [],
            },
        },
    }
    in_nodes3 = [0]
    out_nodes3 = [1]

    # 子网络4（4分类任务）
    node_configs4 = deepcopy(node_configs3)
    node_configs4.update({1: (4, 1, 1, 1)})

    # 子网络5（2分类任务）
    node_configs5 = deepcopy(node_configs3)
    node_configs5.update({1: (2, 1, 1, 1)})

    # 子网络6（3分类任务）
    node_configs6 = deepcopy(node_configs3)
    node_configs6.update({1: (3, 1, 1, 1)})

    # 全局节点映射
    node_mapping = [
        (100, "pre", 0),  # 全局输入：1通道图像
        (101, "pre", 1),  # 全局输入：1通道图像
        (102, "pre", 2),  # 全局输入：1通道图像
        (103, "pre", 3),  # 全局输入：1通道图像
        (104, "pre", 4),  # 全局输入：1通道图像
        (200, "pre", 11),  # 子网络1输出
        (200, "main", 0), # 子网络2输入
        (300, "main", 17),# 子网络2输出，改为节点 17
        (300, "reg1", 0), # 回归任务输入
        (300, "cls1", 0), # 分类任务输入
        (300, "cls2", 0),
        (300, "cls3", 0),
        (300, "cls4", 0),
        (300, "cls5", 0),
        (300, "cls6", 0),
        (300, "cls7", 0),
        (400, "reg1", 1), # 回归任务输出
        (401, "cls1", 1), # 分类任务输出
        (402, "cls2", 1),
        (403, "cls3", 1),
        (404, "cls4", 1),
        (405, "cls5", 1),
        (406, "cls6", 1),
        (407, "cls7", 1),
    ]

    # 子网络实例化
    sub_networks_configs = {
        "pre": (node_configs1, hyperedge_configs1, in_nodes1, out_nodes1),
        "main": (node_configs2, hyperedge_configs2, in_nodes2, out_nodes2),
        "reg1": (node_configs3, hyperedge_configs3, in_nodes3, out_nodes3),
        "cls1": (node_configs4, hyperedge_configs3, in_nodes3, out_nodes3),
        "cls2": (node_configs4, hyperedge_configs3, in_nodes3, out_nodes3),
        "cls3": (node_configs5, hyperedge_configs3, in_nodes3, out_nodes3),
        "cls4": (node_configs5, hyperedge_configs3, in_nodes3, out_nodes3),
        "cls5": (node_configs5, hyperedge_configs3, in_nodes3, out_nodes3),
        "cls6": (node_configs6, hyperedge_configs3, in_nodes3, out_nodes3),
        "cls7": (node_configs5, hyperedge_configs3, in_nodes3, out_nodes3),
    }

    sub_networks = {
        name: HDNet(node_configs, hyperedge_configs, in_nodes, out_nodes, num_dimensions)
        for name, (node_configs, hyperedge_configs, in_nodes, out_nodes) in sub_networks_configs.items()
    }

    # 动态生成配置字典
    config_dict = {
        name: {
            "node_configs": {str(k): list(v) for k, v in node_configs.items()},
            "hyperedge_configs": deepcopy(hyperedge_configs),
            "in_nodes": in_nodes,
            "out_nodes": out_nodes,
        }
        for name, (node_configs, hyperedge_configs, in_nodes, out_nodes) in sub_networks_configs.items()
    }

    # 全局输入输出节点
    in_nodes = [100, 101, 102, 103, 104]
    out_nodes = [400, 401, 402, 403, 404, 405, 406, 407]

    # 外部实例化变换对象
    random_rotate = RandomRotate3D()
    random_flip = RandomFlip3D()
    random_shift = RandomShift3D()
    random_zoom = RandomZoom3D()
    min_max_normalize = MinMaxNormalize3D()
    z_score_normalize = ZScoreNormalize3D()

    # 每个节点的变换列表
    node_transforms = {
        100: [random_rotate, random_flip, random_shift, random_zoom, min_max_normalize, z_score_normalize],
        101: [random_rotate, random_flip, random_shift, random_zoom, min_max_normalize, z_score_normalize],
        102: [random_rotate, random_flip, random_shift, random_zoom, min_max_normalize, z_score_normalize],
        103: [random_rotate, random_flip, random_shift, random_zoom, min_max_normalize, z_score_normalize],
        104: [random_rotate, random_flip, random_shift, random_zoom],
    }

    # 任务与节点映射
    task_node_map = {
        "Feature_1": 401,  # 分类任务，4类
        "Feature_2": 402,  # 分类任务，4类
        "Feature_3": 400,  # 回归任务
        "Feature_4": 403,  # 分类任务，2类
        "Feature_5": 404,  # 分类任务，2类
        "Feature_6": 405,  # 分类任务，2类
        "Feature_7": 406,  # 分类任务，3类
        "Feature_8": 407,  # 分类任务，2类
    }

    cls_features = ['Feature_1', 'Feature_2', 'Feature_4', 'Feature_5', 'Feature_6', 'Feature_7', 'Feature_8']
    reg_features = ['Feature_3']
    lambda_weights = {'cls': [0.4, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05], 'reg': [0.1]}
    focal_params = {
        'Feature_1': ([1.0, 1.0, 1.0, 1.0], 0),
        'Feature_2': ([1.0, 1.0, 1.0, 1.0], 0),
        'Feature_4': ([1.0, 1.0], 0),
        'Feature_5': ([1.0, 1.0], 0),
        'Feature_6': ([1.0, 1.0], 0),
        'Feature_7': ([1.0, 1.0, 1.0], 0),
        'Feature_8': ([1.0, 1.0], 0),
    }

    # 输入节点图像后缀映射
    node_image_mappings = {
        100: ["0000"],
        101: ["0001"],
        102: ["0002"],
        103: ["0003"],
        104: ["0004"],
    }

    # 创建所有输入节点的数据集
    datasets = {
        node: NodeDataset(
            image_dir,
            label_dir,
            node_image_mappings[node],
            node_transforms[node],
            cls_features,
            reg_features,
        )
        for node in in_nodes
    }

    # 获取所有节点的共同 case_ids
    all_case_ids = set.intersection(*(set(datasets[node].case_ids) for node in in_nodes))
    if not all_case_ids:
        raise ValueError("No common case_ids found across all input nodes!")
    all_case_ids = sorted(list(all_case_ids))

    # K折交叉验证
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(all_case_ids)):
        print(f"Fold {fold + 1}")

        # 获取训练和验证的 case_ids
        train_case_ids = [all_case_ids[idx] for idx in train_ids]
        val_case_ids = [all_case_ids[idx] for idx in val_ids]
        split_info = {
            "fold": fold + 1,
            "train_case_ids": train_case_ids,
            "val_case_ids": val_case_ids,
            "train_count": len(train_case_ids),
            "val_count": len(val_case_ids),
        }
        split_save_path = os.path.join(save_dir, f"fold_{fold + 1}_split.json")
        with open(split_save_path, "w") as f:
            json.dump(split_info, f, indent=4)
        print(f"Data split saved to {split_save_path}")

        # 为每个节点创建训练和验证的索引
        dataloaders_train = {}
        dataloaders_val = {}
        for node in in_nodes:
            node_case_ids = datasets[node].case_ids
            train_indices = [
                node_case_ids.index(case_id)
                for case_id in train_case_ids
                if case_id in node_case_ids
            ]
            val_indices = [
                node_case_ids.index(case_id)
                for case_id in val_case_ids
                if case_id in node_case_ids
            ]

            train_subsampler = torch.utils.data.SubsetRandomSampler(train_indices)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_indices)
            dataloaders_train[node] = DataLoader(
                datasets[node],
                batch_size=batch_size,
                sampler=train_subsampler,
                num_workers=16,
            )
            dataloaders_val[node] = DataLoader(
                datasets[node],
                batch_size=batch_size,
                sampler=val_subsampler,
                num_workers=16,
            )

        # 模型、优化器、调度器
        model = MHDNet(
            sub_networks=sub_networks,
            node_mapping=node_mapping,
            in_nodes=in_nodes,
            out_nodes=out_nodes,
            num_dimensions=num_dimensions,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / num_epochs) ** 0.9)

        # 损失函数
        criterion_cls = (
            lambda output, target, feature: focal_loss(
                output,
                target,
                alpha=focal_params[feature][0],
                gamma=focal_params[feature][1],
            )
        )
        criterion_reg = mse_loss

        # 早停
        best_val_loss = float("inf")
        epochs_no_improve = 0
        log = {"fold": fold + 1, "epochs": []}

        # 训练循环
        for epoch in range(num_epochs):
            train_loss, train_cls_loss, train_reg_loss = train(
                model,
                dataloaders_train,
                optimizer,
                criterion_cls,
                criterion_reg,
                lambda_weights,
                cls_features,
                reg_features,
                task_node_map,
                out_nodes,
                epoch,
                num_epochs,
            )

            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_cls_loss": train_cls_loss,
                "train_reg_loss": train_reg_loss,
            }

            if (epoch + 1) % validation_interval == 0:
                val_loss, val_cls_loss, val_reg_loss, metrics = validate(
                    model,
                    dataloaders_val,
                    criterion_cls,
                    criterion_reg,
                    lambda_weights,
                    cls_features,
                    reg_features,
                    task_node_map,
                    out_nodes,
                    epoch,
                    num_epochs,
                    focal_params
                )

                epoch_log.update(
                    {
                        "val_loss": val_loss,
                        "val_cls_loss": val_cls_loss,
                        "val_reg_loss": val_reg_loss,
                        "metrics": metrics,
                    }
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    save_path = os.path.join(save_dir, f"model_fold{fold + 1}_best.pth")
                    torch.save(model.state_dict(), save_path)
                    config = {
                        "sub_networks": config_dict,
                        "node_mapping": node_mapping,
                        "in_nodes": in_nodes,
                        "out_nodes": out_nodes,
                        "num_dimensions": num_dimensions,
                        "task_node_map": task_node_map,
                        "cls_features": cls_features,
                        "reg_features": reg_features,
                        "lambda_weights": lambda_weights,
                        "focal_params": focal_params,
                        "batch_size": batch_size,
                        "num_epochs": num_epochs,
                        "learning_rate": learning_rate,
                        "k_folds": k_folds,
                        "validation_interval": validation_interval,
                        "patience": patience,
                        "node_image_mappings": node_image_mappings,
                        "node_transforms": {
                            str(k): [t.__class__.__name__ for t in v]
                            for k, v in node_transforms.items()
                        },
                    }
                    config_save_path = os.path.join(
                        save_dir, f"model_config_fold{fold + 1}.json"
                    )
                    with open(config_save_path, "w") as f:
                        json.dump(config, f, indent=4)
                    print(f"Model saved to {save_path}, Config saved to {config_save_path}")
                else:
                    epochs_no_improve += validation_interval
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            scheduler.step()
            log["epochs"].append(epoch_log)

        log_save_path = os.path.join(save_dir, f"training_log_fold{fold + 1}.json")
        with open(log_save_path, "w") as f:
            json.dump(log, f, indent=4)
        print(f"Training log saved to {log_save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
